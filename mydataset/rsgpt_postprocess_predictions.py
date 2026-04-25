"""
postprocess_predictions.py

对生成的 predictions.jsonl 做后处理：
  1. 去除 Markdown 格式符号（**bold**、* bullet、# header）
  2. 将 bullet point 列表转换为流畅散文
  3. 控制长度（截断到合理范围）

用法：
  python mydataset/rsgpt_postprocess_predictions.py \
      --input  /opt/data/private/qwen3-vl-master/qwen3-vl/dataset/rsgpt_predictions.jsonl \
      --output /opt/data/private/qwen3-vl-master/qwen3-vl/dataset/rsgpt_predictions_clean.jsonl
"""

import re
import json
import argparse
from pathlib import Path


def remove_markdown(text: str) -> str:
    """去除 Markdown 格式符号，转换为纯文本"""

    # 去除 **bold** 和 *italic*（保留文字内容）
    text = re.sub(r'\*{1,3}([^*\n]+)\*{1,3}', r'\1', text)

    # 去除 # 标题
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # 去除下划线加粗 __text__
    text = re.sub(r'_{1,2}([^_\n]+)_{1,2}', r'\1', text)

    return text


def bullets_to_prose(text: str) -> str:
    """
    将 bullet point 列表转换为散文句子。

    策略：
      - 每个 bullet item 变成独立句子
      - 去掉 bullet 符号（*、-、•）和缩进
      - 多个短句合并（避免碎片化）
    """
    lines = text.split('\n')
    prose_parts = []
    current_sentence = ""

    for line in lines:
        line = line.strip()
        if not line:
            # 空行：如果有积累的句子，加入
            if current_sentence:
                prose_parts.append(current_sentence.strip())
                current_sentence = ""
            continue

        # 检测是否是 bullet 行
        bullet_match = re.match(r'^[-*•·]\s+(.+)', line)
        numbered_match = re.match(r'^\d+[.)]\s+(.+)', line)

        if bullet_match:
            content = bullet_match.group(1).strip()
            # 确保句子以句号结尾
            if content and not content[-1] in '.!?':
                content += '.'
            if current_sentence:
                prose_parts.append(current_sentence.strip())
                current_sentence = ""
            prose_parts.append(content)
        elif numbered_match:
            content = numbered_match.group(1).strip()
            if content and not content[-1] in '.!?':
                content += '.'
            if current_sentence:
                prose_parts.append(current_sentence.strip())
                current_sentence = ""
            prose_parts.append(content)
        else:
            # 普通文本行
            if current_sentence:
                current_sentence += " " + line
            else:
                current_sentence = line

    if current_sentence:
        prose_parts.append(current_sentence.strip())

    return ' '.join(prose_parts)


def clean_whitespace(text: str) -> str:
    """清理多余空白和重复标点"""
    # 多个空格 → 单空格
    text = re.sub(r' {2,}', ' ', text)
    # 多个换行 → 单空格
    text = re.sub(r'\n+', ' ', text)
    # 句号后多个空格
    text = re.sub(r'\.{2,}', '.', text)
    # 去掉行首行尾空白
    text = text.strip()
    return text


def truncate_to_sentences(text: str, max_sentences: int = 6) -> str:
    """
    截断到最多 max_sentences 句，保持完整句子。
    RSGPT 参考描述通常 2-5 句。
    """
    # 按句号/问号/感叹号分割
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= max_sentences:
        return text

    # 取前 max_sentences 句
    return ' '.join(sentences[:max_sentences])


def process_prediction(text: str) -> str:
    """完整的后处理流程"""
    # Step1：去除 Markdown 格式
    text = remove_markdown(text)
    # Step2：bullet → 散文
    text = bullets_to_prose(text)
    # Step3：清理空白
    text = clean_whitespace(text)
    # Step4：截断句数
    text = truncate_to_sentences(text, max_sentences=6)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="原始 predictions.jsonl")
    parser.add_argument("--output", required=True, help="清理后的 predictions.jsonl")
    parser.add_argument("--prediction_key", default="prediction")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records   = []
    changed   = 0

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            original = item.get(args.prediction_key, "")
            cleaned  = process_prediction(original)
            if cleaned != original:
                changed += 1
            item[args.prediction_key] = cleaned
            records.append(item)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"处理完成: {len(records)} 条, 其中 {changed} 条有修改")
    print(f"输出: {output_path}")

    # 打印前2条对比
    with open(input_path, encoding="utf-8") as f:
        originals = [json.loads(l)[args.prediction_key] for l in f if l.strip()][:2]
    with open(output_path, encoding="utf-8") as f:
        cleaneds  = [json.loads(l)[args.prediction_key] for l in f if l.strip()][:2]

    for i, (orig, clean) in enumerate(zip(originals, cleaneds)):
        print(f"\n── 样本{i+1} 原始（前200字）──")
        print(orig[:200])
        print(f"── 样本{i+1} 清理后 ──")
        print(clean[:300])


if __name__ == "__main__":
    main()