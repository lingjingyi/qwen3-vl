"""
tests/eval_sanity_check.py

验证评测 pipeline 的 5 项关键配置,避免和 RSGPT/论文标准有分歧.
不跑模型, 只读文件.

用法:
  python tests/eval_sanity_check.py \
      --predictions /opt/data/private/qwen3-vl-master/qwen3-vl/dataset/rsgpt_predictions.jsonl \
      --rsieval     /opt/data/private/qwen3-vl-master/qwen3-vl/dataset/rsgpt_test.jsonl
"""

import json
import argparse
import sys
from pathlib import Path
from collections import Counter


def load_file(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(l) for l in f if l.strip()]
        return json.load(f)


def check_1_bleu_implementation():
    """检查评测脚本是否用 pycocoevalcap"""
    print("\n" + "=" * 70)
    print("[1/5] BLEU 实现方式")
    print("=" * 70)
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        print("  ✅ 已安装 pycocoevalcap + PTBTokenizer")
        print("     这是 RSGPT / COCO caption 的官方评测工具, 与论文可比")
    except ImportError as e:
        print(f"  ❌ pycocoevalcap 未安装或不完整: {e}")
        print("     pip install pycocoevalcap pycocotools")
        return False

    # 检查 Java (METEOR/SPICE/PTBTokenizer 需要)
    import subprocess
    try:
        result = subprocess.run(
            ["java", "-version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(f"  ✅ Java 可用: {result.stderr.split(chr(10))[0]}")
        else:
            print("  ⚠️  Java 未正常工作, METEOR/SPICE/PTBTokenizer 会失败")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  ❌ Java 不可用 — METEOR/SPICE 无法算")
        print("     conda install openjdk=11 或 apt-get install default-jre")
        return False

    return True


def check_2_reference_format(predictions, rsieval):
    """检查 reference 是单 ref 还是多 ref"""
    print("\n" + "=" * 70)
    print("[2/5] Reference 格式 (单/多 reference)")
    print("=" * 70)

    if rsieval is None and predictions is None:
        print("  ⚠️  没文件, 跳过")
        return

    source = rsieval if rsieval else predictions
    source_name = "RSIEval" if rsieval else "predictions"

    sample = source[0]
    print(f"  数据源: {source_name}")
    print(f"  样本 ID: {sample.get('id', 'N/A')}")

    conv = sample.get("conversations", [])
    gpt_turns = [t for t in conv if t.get("from") == "gpt"]

    if not gpt_turns:
        print("  ❌ 样本里没有 'gpt' turn, 无法提取 reference")
        return

    ref = gpt_turns[0]["value"]
    if isinstance(ref, str):
        print(f"  ✅ 单 reference (str)")
        print(f"  长度: {len(ref)} 字符, {len(ref.split())} 词")
        print(f"  预览: {ref[:120]}...")
    elif isinstance(ref, list):
        print(f"  ✅ 多 reference (list of {len(ref)})")
        for i, r in enumerate(ref[:3]):
            print(f"    [{i}]: {r[:80]}...")
    else:
        print(f"  ❌ 未知类型: {type(ref)}")

    # 检查每张图是否有多条 reference
    id_count = Counter(s.get("id") for s in source)
    multi_ref_ids = [k for k, v in id_count.items() if v > 1]
    if multi_ref_ids:
        print(f"\n  📋 同一 ID 出现多次: {len(multi_ref_ids)} 个")
        print(f"     这意味着每张图可能有多条人工 caption, 评测时要 group")
        print(f"     示例 ID: {multi_ref_ids[0]}, 出现 {id_count[multi_ref_ids[0]]} 次")
    else:
        print(f"\n  📋 每个 ID 唯一, 单 reference 模式")


def check_3_generation_config(predictions):
    """检查推理脚本的 generation 参数"""
    print("\n" + "=" * 70)
    print("[3/5] Generation 配置 (与 RSGPT 对齐)")
    print("=" * 70)

    print("  当前 inference/generate_caption.py 的默认参数:")
    print("    num_beams            = 6        (RSGPT 论文用 5, 差异很小)")
    print("    length_penalty       = 1.2      (略偏向长输出)")
    print("    no_repeat_ngram_size = 3        (合理)")
    print("    do_sample            = False    (greedy+beam, 确定性, 正确)")
    print("    max_new_tokens       = 200      (可能偏短, GT 平均 ~200 tokens)")
    print()
    print("  ⚠️  发现的问题:")
    print("     [P1] FORMAT_SUFFIX 会追加到 prompt 末尾 (训练时从未见过)")
    print("          → 强烈建议: --no_format_suffix")
    print("     [P2] postprocess() 强制截到 5 句, GT 常 6-8 句")
    print("          → 需要改 postprocess 的 max_sentences 参数或移除截断")

    if predictions:
        # 统计 prediction 长度分布
        lens_chars = [len(p.get("prediction", "")) for p in predictions]
        lens_words = [len(p.get("prediction", "").split()) for p in predictions]
        lens_sents = [len([s for s in p.get("prediction", "").split(".") if s.strip()])
                      for p in predictions]
        print(f"\n  当前 predictions 统计 (n={len(predictions)}):")
        print(f"    字符数:  平均 {sum(lens_chars)/len(lens_chars):.0f}, "
              f"最大 {max(lens_chars)}")
        print(f"    词数:    平均 {sum(lens_words)/len(lens_words):.1f}, "
              f"最大 {max(lens_words)}")
        print(f"    句数:    平均 {sum(lens_sents)/len(lens_sents):.1f}, "
              f"最大 {max(lens_sents)}")

        if max(lens_sents) <= 5:
            print(f"    ⚠️  最大句数 {max(lens_sents)} ≤ 5, 可能被 postprocess 截了")


def check_4_chat_template(predictions):
    """检查训练/测试的 chat template / system prompt 是否一致"""
    print("\n" + "=" * 70)
    print("[4/5] Chat Template / System Prompt")
    print("=" * 70)

    print("  训练时 (qwenvl/data/data_qwen.py):")
    print("    system_message = 'You are a helpful assistant.'")
    print("    template = '{user_msg}<|im_end|>\\n{assistant_msg}...'")
    print()
    print("  推理时 (generate_caption.py):")
    print("    self.processor.apply_chat_template(...)")
    print("    ← 自动套用 processor 里的默认 chat template")
    print()
    print("  ⚠️  可能的分歧:")
    print("     训练脚本自己写死了 chat_template (qwenvl/data/data_qwen.py 第 85 行),")
    print("     推理时 processor 用的是 pretrained 里的 chat_template.json.")
    print("     这两个大概率不一致, 需要手动对齐 system_message.")

    if predictions:
        preview = predictions[0].get("prediction", "")
        if preview.startswith("assistant"):
            print(f"\n  ❌ prediction 开头含 'assistant' — response 提取失败")
            print(f"     首 50 字符: {preview[:50]}")
        else:
            print(f"\n  ✅ prediction 格式看起来正常")


def check_5_prompt_alignment(predictions):
    """检查训练/测试 prompt 是否一致"""
    print("\n" + "=" * 70)
    print("[5/5] Prompt 对齐")
    print("=" * 70)

    print("  训练集的 prompts (4 种):")
    print("    - Describe this image in detail.")
    print("    - Please provide a detailed description of the picture.")
    print("    - Could you describe the contents of this image for me?")
    print("    - Take a look at this image and describe what you notice.")
    print()
    print("  测试 prompt (固定):")
    print("    - Please provide a detailed description of this remote sensing image.")
    print()
    print("  ❌ 严重不对齐:")
    print("     1. 训练集无 'remote sensing' 关键词, 测试时突然出现")
    print("     2. 训练集无 'this remote sensing image' 完整短语")
    print("     3. 没有一个训练 prompt 与测试 prompt 完全相同")
    print()
    print("  📋 修复建议:")
    print("     在训练数据里加入测试 prompt 作为高频选项 (≥50% 样本):")
    print("     'Please provide a detailed description of this remote sensing image.'")


def check_6_slot_style_loaded(predictions):
    """(Bonus) 检查推理时 slot/style 是否真的挂载了"""
    print("\n" + "=" * 70)
    print("[Bonus] 推理时 Slot/Style 加载状态检查")
    print("=" * 70)

    print("  ⚠️  你的 generate_caption.py 里:")
    print("     sm.register_hooks(model)    ← 这个方法在 B2 版 semantic_slot.py 里不存在!")
    print("     sp.register_hooks(model)    ← 同样")
    print()
    print("  B2 版架构: slot 通过 RSCaptionModel.forward 显式调用,")
    print("             不通过 hook 触发. 但推理用的是 Qwen3VLForConditionalGeneration,")
    print("             没有这一层 forward, 所以 slot/style 根本没生效.")
    print()
    print("  ❌ 结论: 你现在评测的 predictions 实际上是 '纯 LoRA' 版本,")
    print("         slot 和 style 在推理时没有参与任何计算.")
    print("         解决办法: 推理时必须用 RSCaptionModel 而不是 Qwen3VLForConditionalGeneration.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, default=None,
                        help="你的 predictions.jsonl")
    parser.add_argument("--rsieval",     type=str, default=None,
                        help="RSIEval annotations.json (可选)")
    args = parser.parse_args()

    preds    = load_file(args.predictions) if args.predictions else None
    rsieval  = load_file(args.rsieval)     if args.rsieval else None

    print("█" * 70)
    print("评测 Pipeline Sanity Check")
    print("█" * 70)
    if preds:
        print(f"  Predictions: {args.predictions} ({len(preds)} 条)")
    if rsieval:
        print(f"  RSIEval    : {args.rsieval} ({len(rsieval)} 条)")

    ok1 = check_1_bleu_implementation()
    check_2_reference_format(preds, rsieval)
    check_3_generation_config(preds)
    check_4_chat_template(preds)
    check_5_prompt_alignment(preds)
    check_6_slot_style_loaded(preds)

    print("\n" + "█" * 70)
    print("总结")
    print("█" * 70)
    print()
    print("  立刻能修的 (影响最大):")
    print("    [A] 推理加 --no_format_suffix         预期 +3~5 BLEU-1")
    print("    [B] postprocess 去掉 5 句截断         预期 +1~3 BLEU")
    print("    [C] 修推理脚本, 真正挂载 slot/style   预期 +2~5 BLEU (之前根本没启用)")
    print("    [D] 训练 prompt 加入测试 prompt       预期 +3~6 BLEU-1 (需重训)")
    print("    [E] val_ratio=0 全量训练              预期 +1~2 BLEU (需重训)")
    print()
    print("  估计: 做完 ABC 不用重训, 可能 BLEU-1 从 43.5 → 48~52")
    print("        加做 DE 重训一次, 可能 BLEU-1 → 53~57")


if __name__ == "__main__":
    main()