import os
import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64
from collections.abc import Sequence
import threading

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader
import transformers

from . import data_list
from .rope2d import get_rope_index_3

IGNORE_INDEX        = -100
IMAGE_TOKEN_INDEX   = 151655
VIDEO_TOKEN_INDEX   = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


# ══════════════════════════════════════════════════════════════════════
# 进程级图片缓存（LRU）
# ══════════════════════════════════════════════════════════════════════
from collections import OrderedDict


class ImageCache:
    """线程安全的 LRU 图片缓存（进程级单例）"""

    def __init__(self, maxsize: int = 6000):
        self._cache: OrderedDict = OrderedDict()
        self._lock   = threading.Lock()
        self.maxsize = maxsize
        self.hits    = 0
        self.misses  = 0

    def get(self, key: str):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                return self._cache[key]
            self.misses += 1
            return None

    def put(self, key: str, value):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.maxsize:
                    evicted_key, _ = self._cache.popitem(last=False)
                    logging.debug(f"[ImageCache] 淘汰: {evicted_key}")
                self._cache[key] = value

    def stats(self) -> str:
        total = self.hits + self.misses
        rate  = self.hits / total * 100 if total > 0 else 0.0
        return (f"ImageCache — hits={self.hits}, misses={self.misses}, "
                f"size={len(self._cache)}/{self.maxsize}, "
                f"hit_rate={rate:.1f}%")


_IMAGE_CACHE = ImageCache(maxsize=8192)


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw: List = [],
    visual_type: str = "image",
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."
    if visual_type not in ["image", "video"]:
        raise ValueError("visual_type must be either 'image' or 'video'")

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )
    tokenizer.chat_template = chat_template

    visual_replicate_index = 0
    grid_thw = grid_thw or []
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []

        result = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}], tokenize=True
        )
        if hasattr(result, "input_ids"):
            input_id = list(result["input_ids"])
        elif isinstance(result, dict):
            input_id = list(result["input_ids"])
        else:
            input_id = list(result)
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role    = conv["role"]
                content = conv["content"]
            except:
                role    = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                visual_tag = f"<{visual_type}>"
                if visual_tag in content:
                    parts = content.split(visual_tag)
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        if visual_replicate_index >= len(grid_thw):
                            raise ValueError(
                                f"Mismatch between visual tags ({len(parts)-1}) "
                                f"and grid_thw entries ({len(grid_thw)})"
                            )
                        replacement = (
                            "<|vision_start|>"
                            + f"<|{visual_type}_pad|>" * grid_thw[visual_replicate_index]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            result = tokenizer.apply_chat_template(conv, tokenize=True)
            if hasattr(result, "input_ids"):
                encode_id = list(result["input_ids"])
            elif isinstance(result, dict):
                encode_id = list(result["input_ids"])
            else:
                encode_id = list(result)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                if len(target_mask) >= 3:
                    target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets   = torch.tensor(targets,   dtype=torch.long)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset      = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")

        self.video_max_total_pixels = getattr(data_args, "video_max_total_pixels", 1664 * 28 * 28)
        self.video_min_total_pixels = getattr(data_args, "video_min_total_pixels", 256  * 28 * 28)

        # Qwen3-VL 专用 rope index
        self.get_rope_index = get_rope_index_3

        list_data_dict = []
        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))

            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")

            for ann in annotations:
                ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")
        random.shuffle(list_data_dict)
        rank0_print("Formatting inputs...Skip in lazy mode")

        self.tokenizer      = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args      = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"]  = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

        # ── 改进：预先深拷贝 processor，避免每次 process_image_unified 都拷贝 ──
        # process_image_unified 中 preprocess() 是无副作用的，复用安全。
        # process_video 有动态修改 max_pixels 的需求，仍保留独立深拷贝（见该方法）。
        self._processor = copy.deepcopy(self.data_args.image_processor)

        # ── 根据实际数据量动态调整缓存上限 ────────────────────────────
        all_image_paths = set()
        for sample in list_data_dict:
            paths = sample.get("images") or sample.get("image") or []
            if not isinstance(paths, list):
                paths = [paths]
            for p in paths:
                if os.path.isabs(p):
                    all_image_paths.add(p)
                else:
                    all_image_paths.add(os.path.join(sample.get("data_path", ""), p))

        dynamic_maxsize = max(int(len(all_image_paths) * 1.2), 1024)
        _IMAGE_CACHE.maxsize = dynamic_maxsize
        self._cache = _IMAGE_CACHE
        rank0_print(
            f"[ImageCache] 唯一图片路径数: {len(all_image_paths)}, "
            f"缓存上限自动设为: {dynamic_maxsize}"
        )

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            return np.array([s["num_tokens"] for s in self.list_data_dict])
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_image_unified(self, image_file: str):
        """
        加载并预处理单张图片，命中缓存直接返回克隆。
        使用预拷贝的 self._processor，避免每次重复 deepcopy 的开销（改进4）。
        """
        cached = self._cache.get(image_file)
        if cached is not None:
            image_tensor, grid_thw = cached
            return image_tensor.clone(), grid_thw

        image = Image.open(image_file).convert("RGB")
        visual_processed = self._processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]

        self._cache.put(image_file, (image_tensor, grid_thw))
        return image_tensor.clone(), grid_thw

    def process_video(self, video_file):
        """视频处理需要动态修改 max_pixels，保留独立的 deepcopy"""
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
        vr           = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps      = vr.get_avg_fps()
        video_length = total_frames / avg_fps
        interval     = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)
        target_frames    = min(max(num_frames_to_sample, video_min_frames), video_max_frames)

        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video     = vr.get_batch(frame_idx).asnumpy()
        fps       = len(frame_idx) / video_length

        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"]  = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels

        video_processed = processor.preprocess(images=None, videos=video, return_tensors="pt")
        video_tensor    = video_processed["pixel_values_videos"]
        grid_thw        = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [
            self.data_args.image_processor.temporal_patch_size / fps
        ] * len(grid_thw)

        return video_tensor, grid_thw, second_per_grid_ts

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3

        for attempt_idx in range(num_base_retries):
            try:
                return self._get_item(i)
            except Exception as e:
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                return self._get_item(next_index)
            except Exception as e:
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)

        return self._get_item(i)

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"

        video = None
        if "image" in sources[0] or "images" in sources[0]:
            image_folder = self.list_data_dict[i]["data_path"]

            if "images" in sources[0]:
                image_files = sources[0]["images"]
            else:
                image_files = sources[0]["image"]

            if not isinstance(image_files, list):
                image_files = [image_files]

            processed_images = []
            processed_grids  = []
            for img_file in image_files:
                full_path = img_file if os.path.isabs(img_file) \
                    else os.path.join(image_folder, img_file)
                img_tensor, img_grid = self.process_image_unified(full_path)
                processed_images.append(img_tensor)
                processed_grids.append(img_grid)

            image    = processed_images
            grid_thw = processed_grids

            conversation    = sources[0]["conversations"]
            num_visual_tags = sum(
                conv["value"].count("<image>") for conv in conversation
            )
            if len(grid_thw) != num_visual_tags:
                raise ValueError(
                    f"Sample {i} validation failed: {num_visual_tags} visual tags "
                    f"vs {len(grid_thw)} grid_thw entries\n"
                    f"Conversation: {conversation}\nImages: {image_files}"
                )

            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size ** 2
                for merged_thw in grid_thw
            ]
            sources   = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer,
                grid_thw=grid_thw_merged, visual_type="image"
            )
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                torch.stack(grid_thw, dim=0),
            )

        elif "video" in sources[0]:
            video_file   = self.list_data_dict[i]["video"]
            video_folder = self.list_data_dict[i]["data_path"]

            if isinstance(video_file, list):
                if len(video_file) > 1:
                    video_file = [os.path.join(video_folder, f) for f in video_file]
                    results    = [self.process_video(f) for f in video_file]
                    video, grid_thw, second_per_grid_ts = zip(*results)
                else:
                    video_file = os.path.join(video_folder, video_file[0])
                    video, grid_thw, second_per_grid_ts = self.process_video(video_file)
                    video = [video]
            else:
                video_file = os.path.join(video_folder, video_file)
                video, grid_thw, second_per_grid_ts = self.process_video(video_file)
                video = [video]

            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw        = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size ** 2
                for merged_thw in grid_thw_merged
            ]
            sources   = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer,
                grid_thw=grid_thw_merged, visual_type="video"
            )
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                video_grid_thw=torch.stack(grid_thw, dim=0),
                second_per_grid_ts=second_per_grid_ts,
            )

        else:
            grid_thw_merged = None
            sources   = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged
            )
            position_ids = (
                torch.arange(0, data_dict["input_ids"].size(1))
                .view(1, -1).unsqueeze(0).expand(3, -1, -1)
            )

        if isinstance(i, int):
            data_dict = dict(
                input_ids    = data_dict["input_ids"][0],
                labels       = data_dict["labels"][0],
                position_ids = position_ids,
            )

        if "image" in self.list_data_dict[i] or "images" in self.list_data_dict[i]:
            data_dict["pixel_values"]   = image
            data_dict["image_grid_thw"] = grid_thw
        elif "video" in self.list_data_dict[i]:
            data_dict["pixel_values_videos"] = video
            data_dict["video_grid_thw"]      = grid_thw

        return data_dict

    def log_cache_stats(self):
        rank0_print(f"[ImageCache] {self._cache.stats()}")


# ══════════════════════════════════════════════════════════════════════
# DataCollator
# ══════════════════════════════════════════════════════════════════════

def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)
    padded = []
    for tensor in tensor_list:
        pad_len = max_length - tensor.shape[2]
        padded.append(torch.nn.functional.pad(tensor, (0, pad_len), "constant", 1))
    return torch.cat(padded, dim=1)


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids    = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels       = torch.nn.utils.rnn.pad_sequence(
            labels,    batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids    = input_ids[:,    : self.tokenizer.model_max_length]
        labels       = labels[:,       : self.tokenizer.model_max_length]
        position_ids = position_ids[:, : self.tokenizer.model_max_length]

        batch = dict(
            input_ids      = input_ids,
            labels         = labels,
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id),
        )

        images = list(itertools.chain(*(
            instance["pixel_values"]
            for instance in instances if "pixel_values" in instance
        )))
        videos = list(itertools.chain(*(
            instance["pixel_values_videos"]
            for instance in instances if "pixel_values_videos" in instance
        )))

        if images:
            concat_images = torch.cat(images, dim=0)
            grid_thw = list(itertools.chain(*(
                instance["image_grid_thw"]
                for instance in instances if "image_grid_thw" in instance
            )))
            grid_thw = torch.stack(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw      = None

        if videos:
            concat_videos  = torch.cat(videos, dim=0)
            video_grid_thw = list(itertools.chain(*(
                instance["video_grid_thw"]
                for instance in instances if "video_grid_thw" in instance
            )))
            video_grid_thw = torch.stack(video_grid_thw, dim=0)
        else:
            concat_videos  = None
            video_grid_thw = None

        batch["pixel_values"]        = concat_images
        batch["image_grid_thw"]      = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"]      = video_grid_thw
        batch["position_ids"]        = position_ids
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        seq_lens        = torch.tensor([0] + [len(s) for s in input_ids], dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids       = torch.cat(input_ids,   dim=0)
        labels          = torch.cat(labels,       dim=0)
        position_ids    = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids      = input_ids.unsqueeze(0),
            labels         = labels.unsqueeze(0),
            attention_mask = cumsum_seq_lens,
            position_ids   = position_ids,
        )

        images = list(itertools.chain(*(
            instance["pixel_values"]
            for instance in instances if "pixel_values" in instance
        )))
        videos = list(itertools.chain(*(
            instance["pixel_values_videos"]
            for instance in instances if "pixel_values_videos" in instance
        )))

        if images:
            concat_images = torch.cat(images, dim=0)
            grid_thw = list(itertools.chain(*(
                instance["image_grid_thw"]
                for instance in instances if "image_grid_thw" in instance
            )))
            grid_thw = torch.stack(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw      = None

        if videos:
            concat_videos  = torch.cat(videos, dim=0)
            video_grid_thw = list(itertools.chain(*(
                instance["video_grid_thw"]
                for instance in instances if "video_grid_thw" in instance
            )))
            video_grid_thw = torch.stack(video_grid_thw, dim=0)
        else:
            concat_videos  = None
            video_grid_thw = None

        batch["pixel_values"]        = concat_images
        batch["image_grid_thw"]      = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"]      = video_grid_thw
        return batch


# ══════════════════════════════════════════════════════════════════════
# 改进：加入训练/验证集划分
# ══════════════════════════════════════════════════════════════════════
def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    val_ratio: float = 0.0,   # ★ 0.1 → 0.0, 全量训练冲分数
    seed: int = 42,
) -> Dict:
    """
    构建训练/验证数据模块.

    val_ratio=0.0 (默认): 全量训练, 不留 val set, 最大化数据利用率.
    val_ratio>0        : 按比例划分 val set (原行为, 用于监控过拟合).
    """
    full_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)

    total = len(full_dataset)

    if val_ratio > 0:
        val_size   = max(1, int(total * val_ratio))
        train_size = total - val_size
        generator  = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=generator
        )
        rank0_print(
            f"[Split] total={total}, train={train_size}, "
            f"val={val_size} (val_ratio={val_ratio})"
        )
    else:
        # ★ 全量训练, 无 val set
        train_dataset = full_dataset
        val_dataset   = None
        rank0_print(f"[Split] total={total}, 全量训练 (val_ratio=0)")

    if data_args.data_flatten:
        data_collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer)
    else:
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,       # None 时 HF Trainer 自动跳过 eval
        data_collator=data_collator,
    )


if __name__ == "__main__":
    pass