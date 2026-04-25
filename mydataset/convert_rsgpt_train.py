#!/usr/bin/env python3
"""
convert_rsgpt.py
将 RSGPT captions.json → 训练代码所需的 JSONL 格式

用法：
  python /opt/data/private/qwen3-vl-master/qwen3-vl/mydataset/convert_rsgpt_train.py \
      --captions  /opt/data/private/qwen3-vl-master/data/RSICap/captions.json \
      --image_dir /opt/data/private/qwen3-vl-master/data/RSICap/images \
      --output    /opt/data/private/qwen3-vl-master/qwen3-vl/dataset/rsgpt_train.jsonl
"""

import json
import argparse
import random
from pathlib import Path


def convert(captions_path: str, image_dir: str, output_path: str, seed: int = 42):
    with open(captions_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    annotations = raw.get("annotations", raw) if isinstance(raw, dict) else raw

    image_dir = Path(image_dir)
    records   = []
    skipped   = 0

    for ann in annotations:
        filename    = ann.get("filename", "")
        text_input  = ann.get("text_input", "Please describe this image.")
        text_output = ann.get("text_output", "")

        if not filename or not text_output:
            skipped += 1
            continue

        record = {
            "images": [str((image_dir / filename).resolve())],
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{text_input}"
                },
                {
                    "from": "gpt",
                    "value": text_output
                }
            ]
        }
        records.append(record)

    random.seed(seed)
    random.shuffle(records)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[convert_rsgpt] 完成:")
    print(f"  输入条数  : {len(annotations)}")
    print(f"  跳过条数  : {skipped}")
    print(f"  输出条数  : {len(records)}")
    print(f"  输出路径  : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions",  required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    convert(args.captions, args.image_dir, args.output, args.seed)