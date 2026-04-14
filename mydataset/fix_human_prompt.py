"""
将训练集 jsonl 中所有 human prompt 替换为与推理时完全一致的结构化版本。
必须在重新训练前运行，否则 train/test prompt 不一致导致 BLEU 偏低。

用法：
    python mydataset/fix_human_prompt.py \
        --input  dataset/train_dataset_augmented.jsonl \
        --output dataset/train_dataset_final.jsonl
"""

import json
import argparse
from pathlib import Path

# ── 与 generate_caption.py 中 STRUCTURED_PROMPT 保持逐字一致 ──────────
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


def build_human_value(num_images: int) -> str:
    """生成 N 个 <image> 标签 + 结构化 prompt"""
    image_tags = "\n".join(["<image>"] * num_images)
    return f"{image_tags}\n{STRUCTURED_PROMPT}"


def main():
    parser = argparse.ArgumentParser(description="统一替换训练集 human prompt")
    parser.add_argument("--input",  type=str, required=True,
                        help="输入 jsonl 文件路径")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 jsonl 文件路径")
    parser.add_argument("--dry_run", action="store_true",
                        help="只打印前3条结果，不写入文件")
    args = parser.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {in_path}")

    changed = 0
    skipped = 0
    total   = 0

    lines_out = []

    with open(in_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item  = json.loads(line)
            total += 1

            # 兼容 images（列表）和 image（单张）字段
            images = item.get("images") or item.get("image") or []
            if not isinstance(images, list):
                images = [images]
            num_images = len(images)

            if num_images == 0:
                skipped += 1
                lines_out.append(json.dumps(item, ensure_ascii=False))
                continue

            # 替换 human 的 value
            replaced = False
            for conv in item.get("conversations", []):
                if conv.get("from") == "human":
                    new_value = build_human_value(num_images)
                    if conv["value"] != new_value:
                        conv["value"] = new_value
                        changed += 1
                    replaced = True
                    break

            if not replaced:
                skipped += 1

            lines_out.append(json.dumps(item, ensure_ascii=False))

    # dry_run 模式：只打印不写文件
    if args.dry_run:
        print("=== DRY RUN — 前3条结果预览 ===")
        for line in lines_out[:3]:
            item = json.loads(line)
            for conv in item.get("conversations", []):
                if conv.get("from") == "human":
                    print(conv["value"][:200], "...\n")
        print(f"共 {total} 条，将替换 {changed} 条（dry_run 模式，未写入）")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        for line in lines_out:
            fout.write(line + "\n")

    print(f"处理完成：共 {total} 条，替换 {changed} 条，跳过 {skipped} 条")
    print(f"输出文件：{out_path}")


if __name__ == "__main__":
    main()