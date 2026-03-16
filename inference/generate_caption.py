import os
import json
import argparse
import logging
from typing import List, Dict
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CaptionGenerator:
    """使用微调后的 Qwen3-VL 模型生成 Caption"""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        """
        初始化生成器

        Args:
            model_path: 模型路径
            device: 设备
            torch_dtype: 数据类型
        """
        self.device = device
        self.torch_dtype = torch_dtype

        logger.info(f"Loading model from {model_path}")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto"
        )

        logger.info("Loading processor")
        self.processor = AutoProcessor.from_pretrained(model_path)

        logger.info("Model loaded successfully")

    def generate_single_caption(
        self,
        image_path: str,
        prompt: str = "请为这张图片生成一个详细的描述。",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        为单张图片生成 Caption

        Args:
            image_path: 图片路径
            prompt: 提示词
            max_new_tokens: 最大生成 token 数
            temperature: 温度
            top_p: top-p 采样
            do_sample: 是否采样

        Returns:
            生成的 Caption
        """
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample
            )

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        assistant_prefix = "assistant\n"
        if assistant_prefix in generated_text:
            generated_text = generated_text.split(assistant_prefix)[-1].strip()

        return generated_text

    def generate_multi_image_caption(
        self,
        image_paths: List[str],
        prompt: str = "请描述这些图片的内容。",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        为多张图片生成 Caption（如变化描述）

        Args:
            image_paths: 图片路径列表
            prompt: 提示词
            max_new_tokens: 最大生成 token 数
            temperature: 温度
            top_p: top-p 采样
            do_sample: 是否采样

        Returns:
            生成的 Caption
        """
        images = [Image.open(p).convert("RGB") for p in image_paths]

        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=text,
            images=images,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample
            )

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        assistant_prefix = "assistant\n"
        if assistant_prefix in generated_text:
            generated_text = generated_text.split(assistant_prefix)[-1].strip()

        return generated_text

    def batch_generate(
        self,
        samples: List[Dict],
        image_key: str = "image",
        id_key: str = "id",
        prompt: str = "请为这张图片生成一个详细的描述。",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        output_file: str = None
    ) -> List[Dict]:
        """
        批量生成 Caption

        Args:
            samples: 样本列表
            image_key: 图片路径在样本中的键名
            id_key: 样本 ID 键名
            prompt: 提示词
            max_new_tokens: 最大生成 token 数
            temperature: 温度
            top_p: top-p 采样
            do_sample: 是否采样
            output_file: 输出文件路径（JSONL）

        Returns:
            生成结果列表
        """
        results = []

        for sample in tqdm(samples, desc="Generating captions"):
            try:
                sample_id = sample.get(id_key, str(len(results)))

                if image_key in sample:
                    image_path = sample[image_key]
                    if isinstance(image_path, list):
                        caption = self.generate_multi_image_caption(
                            image_path,
                            prompt=prompt,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=do_sample
                        )
                    else:
                        caption = self.generate_single_caption(
                            image_path,
                            prompt=prompt,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=do_sample
                        )
                else:
                    logger.warning(f"No image found in sample {sample_id}, skipping")
                    continue

                result = {
                    id_key: sample_id,
                    "prediction": caption,
                    **sample
                }
                results.append(result)

                if output_file:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')

            except Exception as e:
                logger.error(f"Error processing sample {sample}: {e}")
                continue

        return results


def load_samples(file_path: str) -> List[Dict]:
    """加载样本文件"""
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL Caption 生成工具")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型路径"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="输入样本文件 (JSON 或 JSONL)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="predictions.jsonl",
        help="输出结果文件"
    )
    parser.add_argument(
        "--image_key",
        type=str,
        default="image",
        help="图片路径在样本中的键名"
    )
    parser.add_argument(
        "--id_key",
        type=str,
        default="id",
        help="样本 ID 键名"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="请为这张图片生成一个详细的描述。",
        help="提示词"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="最大生成 token 数"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="温度"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="top-p 采样"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=True,
        help="是否采样"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Qwen3-VL Caption Generator")
    logger.info("=" * 60)

    logger.info(f"Loading samples from: {args.input_file}")
    samples = load_samples(args.input_file)
    logger.info(f"Loaded {len(samples)} samples")

    if os.path.exists(args.output_file):
        logger.warning(f"Output file {args.output_file} already exists, will append")

    generator = CaptionGenerator(
        model_path=args.model_path,
        device=args.device
    )

    logger.info("Starting batch generation...")
    results = generator.batch_generate(
        samples=samples,
        image_key=args.image_key,
        id_key=args.id_key,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        output_file=args.output_file
    )

    logger.info(f"Generated {len(results)} captions")
    logger.info(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
