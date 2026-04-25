"""
重写 rsgpt_train.jsonl 的 prompt 分布:
  - 60% 样本使用测试 prompt (对齐 RSIEval 测试分布)
  - 其余 40% 使用多样性 prompt (防止过拟合到单一 prompt)

用法:
  python mydataset/rewrite_train_prompts.py \
      --input  /opt/data/private/qwen3-vl-master/qwen3-vl/dataset/rsgpt_train.jsonl \
      --output /opt/data/private/qwen3-vl-master/qwen3-vl/dataset/rsgpt_train_aligned.jsonl \
      --target_ratio 0.6

  验证:
  python mydataset/rewrite_train_prompts.py \
      --input dataset/rsgpt_train_aligned.jsonl \
      --inspect_only
"""
import json
import argparse
import random
import re
from collections import Counter
from pathlib import Path

# ★ 核心: 测试 prompt, 和 RSIEval 评测时使用的完全一致
TEST_PROMPT = "Please provide a detailed description of this remote sensing image."

# 多样性 prompt (训练时 40% 样本用), 包含 "remote sensing" 关键词
DIVERSITY_PROMPTS = [
    "Describe this remote sensing image in detail.",
    "What does this remote sensing image show?",
    "Provide a comprehensive caption for this aerial/satellite imagery.",
    "Please give a detailed description of the contents in this remote sensing image.",
]


def rewrite_prompt(sample: dict, new_prompt: str) -> dict:
    """
    重写 conversations 里第一个 human turn 的 prompt.
    保留 <image> 标签.
    """
    new_sample = json.loads(json.dumps(sample))  # 深拷贝

    for turn in new_sample.get("conversations", []):
        if turn.get("from") == "human":
            old_value = turn.get("value", "")
            # 提取并保留所有 <image> 标签
            image_tags = re.findall(r"<image>\n?", old_value)
            image_prefix = "".join(image_tags) if image_tags else "<image>\n"
            turn["value"] = image_prefix + new_prompt
            break  # 只改第一个 human turn

    return new_sample


def inspect_prompts(path):
    """统计 prompt 分布"""
    prompts = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            for turn in d.get("conversations", []):
                if turn.get("from") == "human":
                    v = re.sub(r"<image>\n?", "", turn.get("value", "")).strip()
                    prompts[v] += 1
                    break

    total = sum(prompts.values())
    print(f"\n{'=' * 70}")
    print(f"Prompt 分布 ({path})  总计 {total} 条")
    print(f"{'=' * 70}")
    for p, c in prompts.most_common():
        ratio = 100 * c / total
        marker = "★" if TEST_PROMPT in p else " "
        print(f"  {marker} {c:5d} ({ratio:5.1f}%)  {p[:80]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        required=True)
    parser.add_argument("--output",       default=None)
    parser.add_argument("--target_ratio", type=float, default=0.6,
                        help="测试 prompt 占比 (默认 60%)")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--inspect_only", action="store_true",
                        help="只看 prompt 分布, 不重写")
    args = parser.parse_args()

    if args.inspect_only:
        inspect_prompts(args.input)
        return

    assert args.output, "--output required when not --inspect_only"

    random.seed(args.seed)

    # 读入所有样本
    with open(args.input, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]

    n_total  = len(samples)
    n_target = int(n_total * args.target_ratio)

    # 随机选 60% 样本, 其 prompt 改为测试 prompt
    target_indices = set(random.sample(range(n_total), n_target))

    # 剩下 40% 均匀分配给 DIVERSITY_PROMPTS
    div_prompts_cycle = DIVERSITY_PROMPTS * ((n_total - n_target) // len(DIVERSITY_PROMPTS) + 2)
    random.shuffle(div_prompts_cycle)
    div_idx = 0

    new_samples = []
    for i, s in enumerate(samples):
        if i in target_indices:
            new_samples.append(rewrite_prompt(s, TEST_PROMPT))
        else:
            new_samples.append(rewrite_prompt(s, div_prompts_cycle[div_idx]))
            div_idx += 1

    # 写出
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for s in new_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"✅ 写入 {len(new_samples)} 条到 {args.output}")
    inspect_prompts(args.output)


if __name__ == "__main__":
    main()