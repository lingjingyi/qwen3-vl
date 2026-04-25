
"""
convert_rsgpt_test.py
将 RSGPT 测试集 JSON（含 caption + qa_pairs）→ JSONL，只保留 caption 部分

用法：
  python /opt/data/private/qwen3-vl-master/qwen3-vl/mydataset/convert_rsgpt_test.py \
      --input     /opt/data/private/qwen3-vl-master/data/RSIEval/annotations.json \
      --image_dir /opt/data/private/qwen3-vl-master/data/RSIEval/images \
      --output    /opt/data/private/qwen3-vl-master/qwen3-vl/dataset/rsgpt_test.jsonl
"""

import json
import argparse
import os
from pathlib import Path

CAPTION_INSTRUCTION = (
    "Please provide a detailed description of this remote sensing image."
)


def convert(input_path: str, image_dir: str, output_path: str, verify: bool = False):
    # ── 加载原始数据 ─────────────────────────────────────────────────
    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 兼容 {"annotations": [...]} 或直接 [...]
    annotations = raw.get("annotations", raw) if isinstance(raw, dict) else raw

    image_dir   = Path(image_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records       = []
    skipped       = []
    missing_image = []

    for ann in annotations:
        filename = ann.get("filename", "").strip()
        caption  = ann.get("caption",  "").strip()

        # ── 跳过无效条目 ─────────────────────────────────────────────
        if not filename:
            skipped.append({"reason": "empty filename", "ann": ann})
            continue
        if not caption:
            skipped.append({"reason": "empty caption", "filename": filename})
            continue

        abs_image_path = str((image_dir / filename).resolve())

        # ── 图片存在性检查（可选，verify 模式下报告） ────────────────
        if verify and not os.path.exists(abs_image_path):
            missing_image.append(abs_image_path)

        # ── 用 filename 去扩展名作为唯一 id ─────────────────────────
        sample_id = Path(filename).stem

        record = {
            "id": sample_id,
            "images": [abs_image_path],
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{CAPTION_INSTRUCTION}"
                },
                {
                    "from": "gpt",
                    "value": caption
                }
            ]
        }
        records.append(record)

    # ── 写出 JSONL ────────────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── 统计报告 ─────────────────────────────────────────────────────
    print(f"\n[convert_rsgpt_test] 转换完成")
    print(f"  输入条数      : {len(annotations)}")
    print(f"  输出条数      : {len(records)}")
    print(f"  跳过条数      : {len(skipped)}")
    for s in skipped:
        print(f"    - {s}")

    if verify:
        print(f"\n[验证] 图片路径检查:")
        if missing_image:
            print(f"  缺失图片数: {len(missing_image)}")
            for p in missing_image[:10]:
                print(f"    ✗ {p}")
            if len(missing_image) > 10:
                print(f"    ... 共 {len(missing_image)} 张缺失")
        else:
            print(f"  ✅ 全部 {len(records)} 张图片路径均存在")

        # ── 抽样展示前2条，方便人工核查 ─────────────────────────────
        print(f"\n[验证] 前2条样本预览:")
        for r in records[:2]:
            print(json.dumps(r, ensure_ascii=False, indent=2))

    print(f"\n  输出路径      : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert RSGPT test JSON to JSONL for generate.py + evaluate.py"
    )
    parser.add_argument("--input",     required=True, help="RSGPT 测试集 JSON 路径")
    parser.add_argument("--image_dir", required=True, help="图像根目录路径")
    parser.add_argument("--output",    required=True, help="输出 JSONL 路径")
    parser.add_argument("--verify",    action="store_true",
                        help="检查图片路径是否存在并打印样本预览")
    args = parser.parse_args()

    convert(args.input, args.image_dir, args.output, args.verify)