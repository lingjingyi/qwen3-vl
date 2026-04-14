import os
import re
import json
import math
import argparse
import logging
from typing import List, Dict, Optional
from pathlib import Path
from collections import Counter

import torch
from PIL import Image
from tqdm import tqdm

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# 结构化 Prompt — 与训练数据 human prompt 逐字一致
# ══════════════════════════════════════════════════════════════════════
STRUCTURED_PROMPT = (
    "Please generate a detailed description for these images. "
    "Your response must be written in flowing prose (no bullet points or numbered lists) "
    "and cover the following aspects in order:\n"
    "1) Identify the scene type (cultural, sports, academic, residential, or campus service) "
    "and state the total number of buildings and their layout pattern "
    "(e.g. enclosed layout, stacked pattern, column pattern, U-shaped layout, paired arrangement).\n"
    "2) For each main building, describe its position (left/center/right/upper/lower), "
    "shape (e.g. irregular polygon, roughly rectangular, L-shaped), "
    "scale (small/medium/large), and floor count (low-rise/mid-rise).\n"
    "3) For each main building, describe the roof: color, type (flat/gabled/hipped/sloped), "
    "and any rooftop equipment (solar panels, ventilation units, HVAC units, "
    "chimneys, dormers, rooftop equipment).\n"
    "4) Describe the surrounding environment: "
    "roads (paved roads/pathways), parking (parking areas/parking lots/parking spaces), "
    "and greenery (green lawns/trees/dense green vegetation).\n"
    "Keep the total response to 3 to 5 sentences."
)

# ══════════════════════════════════════════════════════════════════════
# 候选评分常量
# ══════════════════════════════════════════════════════════════════════
TARGET_SENTENCE_MIN = 3
TARGET_SENTENCE_MAX = 5
TARGET_WORD_MID     = 70
TARGET_WORD_MIN     = 35
TARGET_WORD_MAX     = 130

# 从训练数据分布中提取的参考 bigram 向量
# 反映任务领域的高频词对，权重越高说明越接近训练分布
# 与 CIDEr 逻辑一致：领域特有词权重高，通用词权重低
REFERENCE_BIGRAMS = Counter({
    "flat roof":           3,
    "low-rise":            3,
    "paved roads":         2,
    "parking lots":        2,
    "irregular polygon":   2,
    "rooftop equipment":   2,
    "ventilation units":   2,
    "standalone building": 2,
    "open ground":         2,
    "roughly rectangular": 1,
    "enclosed layout":     1,
    "scattered greenery":  1,
    "solar panels":        1,
    "hvac units":          1,
    "mid-rise":            1,
    "parking areas":       1,
    "gabled roof":         1,
    "sloped roof":         1,
})


def _count_sentences(text: str) -> int:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return len([p for p in parts if p.strip()])


def _count_words(text: str) -> int:
    return len(text.split())


def _ngram_tfidf_vector(text: str, n: int = 2) -> Counter:
    """
    提取 n-gram 并按词频加权。
    CIDEr 核心思想：高频领域词权重高，通用词权重低。
    此处用训练描述的高频词作为近似 IDF 权重基准。
    """
    tokens = text.lower().split()
    ngrams = [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return Counter(ngrams)


def _cosine_sim(a: Counter, b: Counter) -> float:
    """计算两个 Counter 向量的余弦相似度"""
    keys = set(a) | set(b)
    dot  = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na   = math.sqrt(sum(v ** 2 for v in a.values())) + 1e-9
    nb   = math.sqrt(sum(v ** 2 for v in b.values())) + 1e-9
    return dot / (na * nb)


def _score_candidate(text: str) -> float:
    """
    对 beam search 候选进行综合评分，选出最符合训练数据分布的候选。
    评分维度：
      - 句数是否在目标范围（权重最高）
      - 词数是否在目标范围
      - 词数接近中位数的程度
      - 与参考 bigram 分布的余弦相似度（近似 CIDEr 逻辑）
    """
    n_sent  = _count_sentences(text)
    n_words = _count_words(text)

    sent_ok   = 1.0 if TARGET_SENTENCE_MIN <= n_sent  <= TARGET_SENTENCE_MAX else 0.0
    word_ok   = 1.0 if TARGET_WORD_MIN     <= n_words <= TARGET_WORD_MAX     else 0.0
    word_prox = max(0.0, 1.0 - abs(n_words - TARGET_WORD_MID) / TARGET_WORD_MID)

    cand_vec  = _ngram_tfidf_vector(text, n=2)
    cider_sim = _cosine_sim(cand_vec, REFERENCE_BIGRAMS)

    return sent_ok * 2.0 + word_ok * 1.0 + word_prox * 0.5 + cider_sim * 1.5


class CaptionGenerator:
    """使用微调后的 Qwen3-VL 模型生成结构化 Caption（Beam Search + Batch 推理版）"""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        logger.info(f"Loading model from {model_path}")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        logger.info("Model loaded successfully")

    # ──────────────────────────────────────────────────────────────────
    # 单样本 inputs 构建
    # ──────────────────────────────────────────────────────────────────
    def _build_inputs(
        self, images: List[Image.Image], prompt: str
    ) -> Dict:
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.processor(
            text=text, images=images, return_tensors="pt"
        ).to(self.model.device)

    # ──────────────────────────────────────────────────────────────────
    # Batch inputs 构建（核心改进）
    # 任务图像均为 512×512，token 长度高度一致，padding 损耗极小
    # ──────────────────────────────────────────────────────────────────
    def _build_inputs_batch(
        self, batch_images: List[List[Image.Image]], prompt: str
    ) -> Dict:
        """
        将多个样本打包为一个 batch。
        batch_images: [[img1,...,img6], [img1,...,img6], ...]
                      外层对应不同样本，内层对应每个样本的多张图片（1张卫星图+5张侧视图）
        """
        texts      = []
        all_images = []

        for images in batch_images:
            content = [{"type": "image", "image": img} for img in images]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)
            all_images.extend(images)

        return self.processor(
            text=texts,
            images=all_images,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

    def _extract_response(self, full_text: str) -> str:
        prefix = "assistant\n"
        if prefix in full_text:
            return full_text.split(prefix)[-1].strip()
        return full_text.strip()

    # ──────────────────────────────────────────────────────────────────
    # Beam Search 解码（单样本，返回多候选供筛选）
    # ──────────────────────────────────────────────────────────────────
    def _decode_beam(
        self,
        inputs: Dict,
        max_new_tokens: int,
        min_new_tokens: int,
        num_beams: int,
        length_penalty: float,
        no_repeat_ngram_size: int,
        num_return_sequences: int,
    ) -> List[str]:
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                num_return_sequences=num_return_sequences,
                do_sample=False,
                early_stopping=True,
            )
        raw_texts = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [self._extract_response(t) for t in raw_texts]

    # ──────────────────────────────────────────────────────────────────
    # Batch Beam Search 解码（多样本，每样本返回 1 条最优候选）
    # num_return_sequences 固定为 1 以支持 padding batch 推理
    # 512×512 固定分辨率下各样本 token 长度一致，padding 损耗极小
    # ──────────────────────────────────────────────────────────────────
    def _decode_beam_batch(
        self,
        inputs: Dict,
        max_new_tokens: int,
        min_new_tokens: int,
        num_beams: int,
        length_penalty: float,
        no_repeat_ngram_size: int,
    ) -> List[str]:
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                num_return_sequences=1,
                do_sample=False,
                early_stopping=True,
            )
        raw_texts = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [self._extract_response(t) for t in raw_texts]

    def _best_candidate(self, candidates: List[str]) -> str:
        return max(candidates, key=_score_candidate)

    # ──────────────────────────────────────────────────────────────────
    # 单条推理对外接口
    # ──────────────────────────────────────────────────────────────────
    def generate_caption(
        self,
        image_paths,
        prompt: str = STRUCTURED_PROMPT,
        max_new_tokens: int = 256,
        min_new_tokens: int = 40,
        num_beams: int = 8,
        length_penalty: float = 1.5,
        no_repeat_ngram_size: int = 3,
        num_return_sequences: int = 4,
    ) -> str:
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self._build_inputs(images, prompt)
        n_ret  = min(num_return_sequences, num_beams)
        candidates = self._decode_beam(
            inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_return_sequences=n_ret,
        )
        return self._best_candidate(candidates)

    # ──────────────────────────────────────────────────────────────────
    # 批量推理对外接口（核心改进：真正的 batch 推理）
    # ──────────────────────────────────────────────────────────────────
    def batch_generate(
        self,
        samples: List[Dict],
        image_key: str = "image",
        id_key: str = "id",
        prompt: str = STRUCTURED_PROMPT,
        max_new_tokens: int = 256,
        min_new_tokens: int = 40,
        num_beams: int = 8,
        length_penalty: float = 1.5,
        no_repeat_ngram_size: int = 3,
        num_return_sequences: int = 4,
        batch_size: int = 4,
        output_file: Optional[str] = None,
    ) -> List[Dict]:
        """
        批量生成描述。
        batch_size=4 在 A800 + 512×512 固定分辨率（6张图/样本）下推理速度约提升 3-4x。

        策略：
          - batch_size > 1：走 batch 路径，num_return_sequences=1，牺牲候选多样性换吞吐量
          - batch_size = 1：走单样本路径，num_return_sequences=4，充分利用 beam 候选筛选
        """
        results = []

        # ── 预处理：提取所有样本的图片路径 ──────────────────────────
        valid_samples = []
        for sample in samples:
            sample_id = sample.get(id_key, str(len(valid_samples)))
            paths = (
                sample.get(image_key)
                or sample.get("images")
                or sample.get("image")
            )
            if paths is None:
                logger.warning(f"样本 {sample_id} 无图片字段，跳过")
                continue
            if not isinstance(paths, list):
                paths = [paths]
            valid_samples.append({**sample, "_paths": paths, id_key: sample_id})

        logger.info(f"有效样本数: {len(valid_samples)}，batch_size={batch_size}")

        # ── 按 batch_size 分组推理 ────────────────────────────────────
        for batch_start in tqdm(
            range(0, len(valid_samples), batch_size), desc="Generating captions"
        ):
            batch = valid_samples[batch_start: batch_start + batch_size]

            try:
                batch_images = [
                    [Image.open(p).convert("RGB") for p in s["_paths"]]
                    for s in batch
                ]

                if len(batch) == 1:
                    # 单样本：走多候选路径，充分利用 beam search 筛选
                    inputs     = self._build_inputs(batch_images[0], prompt)
                    n_ret      = min(num_return_sequences, num_beams)
                    candidates = self._decode_beam(
                        inputs,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        num_beams=num_beams,
                        length_penalty=length_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        num_return_sequences=n_ret,
                    )
                    captions = [self._best_candidate(candidates)]
                else:
                    # 多样本：走 batch 路径，num_return_sequences=1
                    inputs   = self._build_inputs_batch(batch_images, prompt)
                    captions = self._decode_beam_batch(
                        inputs,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        num_beams=num_beams,
                        length_penalty=length_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                    )

                for sample, caption in zip(batch, captions):
                    clean_sample = {k: v for k, v in sample.items() if k != "_paths"}
                    result = {id_key: sample[id_key], "prediction": caption, **clean_sample}
                    results.append(result)
                    if output_file:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")

            except Exception as e:
                logger.error(
                    f"处理 batch [{batch_start}:{batch_start + len(batch)}] 时出错: {e}，逐条重试"
                )
                # batch 失败时逐条重试，保证不丢数据
                for sample in batch:
                    try:
                        images     = [Image.open(p).convert("RGB") for p in sample["_paths"]]
                        inputs     = self._build_inputs(images, prompt)
                        n_ret      = min(num_return_sequences, num_beams)
                        candidates = self._decode_beam(
                            inputs,
                            max_new_tokens=max_new_tokens,
                            min_new_tokens=min_new_tokens,
                            num_beams=num_beams,
                            length_penalty=length_penalty,
                            no_repeat_ngram_size=no_repeat_ngram_size,
                            num_return_sequences=n_ret,
                        )
                        caption      = self._best_candidate(candidates)
                        clean_sample = {k: v for k, v in sample.items() if k != "_paths"}
                        result = {id_key: sample[id_key], "prediction": caption, **clean_sample}
                        results.append(result)
                        if output_file:
                            with open(output_file, "a", encoding="utf-8") as f:
                                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    except Exception as e2:
                        logger.error(f"单条重试样本 {sample[id_key]} 仍失败: {e2}")

        return results


# ══════════════════════════════════════════════════════════════════════
# 文件加载
# ══════════════════════════════════════════════════════════════════════
def load_samples(file_path: str) -> List[Dict]:
    if file_path.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL Caption 生成工具（Beam Search + Batch 推理版，6张512×512输入）"
    )
    parser.add_argument("--model_path",           type=str,   required=True,
                        help="微调后模型路径（merged_model 目录）")
    parser.add_argument("--input_file",           type=str,   required=True,
                        help="测试集文件路径（.json 或 .jsonl）")
    parser.add_argument("--output_file",          type=str,   default="predictions.jsonl")
    parser.add_argument("--image_key",            type=str,   default="image",
                        help="样本中图片路径的字段名（也兼容 images 字段）")
    parser.add_argument("--id_key",               type=str,   default="id")
    parser.add_argument("--prompt",               type=str,   default=None,
                        help="自定义 prompt；不传则使用内置结构化 prompt")
    # ── 生成长度控制 ────────────────────────────────────────────────
    parser.add_argument("--max_new_tokens",       type=int,   default=256)
    parser.add_argument("--min_new_tokens",       type=int,   default=40)
    # ── Beam Search 参数 ────────────────────────────────────────────
    parser.add_argument("--num_beams",            type=int,   default=8)
    parser.add_argument("--length_penalty",       type=float, default=1.5)
    parser.add_argument("--no_repeat_ngram_size", type=int,   default=3)
    parser.add_argument("--num_return_sequences", type=int,   default=4,
                        help="单样本模式下的候选数，须 <= num_beams")
    # ── Batch 推理参数 ───────────────────────────────────────────────
    parser.add_argument("--batch_size",           type=int,   default=4,
                        help="批量推理 batch 大小，6张512×512输入建议 4")
    parser.add_argument("--device",               type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    if args.num_return_sequences > args.num_beams:
        logger.warning(
            f"num_return_sequences({args.num_return_sequences}) > "
            f"num_beams({args.num_beams})，自动修正为 {args.num_beams}"
        )
        args.num_return_sequences = args.num_beams

    logger.info("=" * 60)
    logger.info("Qwen3-VL Caption Generator — Beam Search + Batch")
    logger.info(f"  input          : {args.input_file}")
    logger.info(f"  output         : {args.output_file}")
    logger.info(f"  batch_size     : {args.batch_size}")
    logger.info(f"  max_new_tokens : {args.max_new_tokens}")
    logger.info(f"  min_new_tokens : {args.min_new_tokens}")
    logger.info(f"  num_beams      : {args.num_beams}")
    logger.info(f"  length_penalty : {args.length_penalty}")
    logger.info(f"  no_repeat_ngram: {args.no_repeat_ngram_size}")
    logger.info(f"  num_return_seq : {args.num_return_sequences} (单样本模式)")
    logger.info(f"  prompt         : {'custom' if args.prompt else 'built-in structured'}")
    logger.info("=" * 60)

    samples = load_samples(args.input_file)
    logger.info(f"Loaded {len(samples)} samples")

    if os.path.exists(args.output_file):
        logger.warning(f"输出文件已存在，将追加写入: {args.output_file}")

    generator = CaptionGenerator(
        model_path=args.model_path,
        device=args.device,
    )

    generator.batch_generate(
        samples=samples,
        image_key=args.image_key,
        id_key=args.id_key,
        prompt=args.prompt or STRUCTURED_PROMPT,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        num_return_sequences=args.num_return_sequences,
        batch_size=args.batch_size,
        output_file=args.output_file,
    )

    logger.info(f"Done -> {args.output_file}")


if __name__ == "__main__":
    main()