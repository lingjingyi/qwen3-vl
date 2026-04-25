"""
并排对比多个推理配置的评测结果.
用法:
    python tests/compare_inference_configs.py \
        --reference /opt/data/private/qwen3-vl-master/qwen3-vl/dataset/rsgpt_test.jsonl \
        --predictions /opt/data/private/qwen3-vl-master/qwen3-vl/dataset/pred_A1.jsonl /opt/data/private/qwen3-vl-master/qwen3-vl/dataset/pred_A2.jsonl /opt/data/private/qwen3-vl-master/qwen3-vl/dataset/pred_A3.jsonl \
        --names A1 A2 A3
"""
import json
import argparse
import subprocess
import os
from pathlib import Path


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def quick_length_stats(pred_file):
    preds = load_jsonl(pred_file)
    word_counts = [len(p.get("prediction", "").split()) for p in preds]
    sent_counts = [
        len([s for s in p.get("prediction", "").split(".") if s.strip()])
        for p in preds
    ]
    return {
        "n": len(preds),
        "words_avg": sum(word_counts) / len(word_counts),
        "words_min": min(word_counts),
        "words_max": max(word_counts),
        "sents_avg": sum(sent_counts) / len(sent_counts),
    }


def run_eval(pred_file, ref_file, out_file):
    """调你现有的评测脚本"""
    cmd = [
        "python",
        "evaluation/evaluate_caption2.py",   # ★ 如果你的评测脚本叫别的,改这里
        "--prediction_file", pred_file,
        "--reference_file",  ref_file,
        "--output_file",     out_file,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ❌ 评测失败: {result.stderr[-500:]}")
        return None
    with open(out_file, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference",   required=True)
    parser.add_argument("--predictions", required=True, nargs="+")
    parser.add_argument("--names",       required=True, nargs="+")
    parser.add_argument("--eval_script", default="evaluation/evaluate_caption2.py",
                        help="你的评测脚本路径,默认 evaluation/evaluate_caption2.py")
    args = parser.parse_args()

    assert len(args.predictions) == len(args.names), \
        "predictions 和 names 数量要一致"

    # 允许自定义评测脚本路径
    global _EVAL_SCRIPT
    _EVAL_SCRIPT = args.eval_script

    results = {}
    for name, pred_file in zip(args.names, args.predictions):
        print(f"\n{'='*60}")
        print(f"[{name}] {pred_file}")
        print(f"{'='*60}")

        if not Path(pred_file).exists():
            print(f"  ⚠️  文件不存在, 跳过")
            continue

        # 长度统计
        stats = quick_length_stats(pred_file)
        print(f"  样本数: {stats['n']}")
        print(f"  词数   avg={stats['words_avg']:.1f}  "
              f"min={stats['words_min']}  max={stats['words_max']}")
        print(f"  句数   avg={stats['sents_avg']:.1f}")

        # 跑评测
        out_file = f"/tmp/eval_{name}.json"
        metrics = run_eval(pred_file, args.reference, out_file)
        if metrics:
            results[name] = {"stats": stats, "metrics": metrics}
            for k in ["bleu1", "bleu2", "bleu3", "bleu4",
                     "meteor", "rougeL", "cider", "spice"]:
                if k in metrics:
                    print(f"  {k:8s}: {metrics[k]:.4f}")

    # ── 对比表 ─────────────────────────────────────────
    if not results:
        print("\n无有效结果"); return

    print("\n" + "█" * 80)
    print("对比表")
    print("█" * 80)

    metric_keys = ["bleu1", "bleu2", "bleu3", "bleu4",
                   "meteor", "rougeL", "cider", "spice"]

    # 表头
    header = f"{'metric':<10s}"
    for name in results:
        header += f"  {name:>8s}"
    header += f"  {'best':>6s}"
    print(header)
    print("-" * len(header))

    best_name, best_bleu1 = None, -1
    for k in metric_keys:
        row = f"{k:<10s}"
        max_val, max_name = -1, ""
        for name, r in results.items():
            v = r["metrics"].get(k, -1)
            row += f"  {v:>8.4f}"
            if v > max_val:
                max_val, max_name = v, name
        row += f"  {max_name:>6s}"
        print(row)

        if k == "bleu1":
            best_bleu1, best_name = max_val, max_name

    # 长度对比
    print()
    lr = f"{'words':<10s}"
    for name, r in results.items():
        lr += f"  {r['stats']['words_avg']:>8.1f}"
    print(lr)

    print()
    print(f"🏆 BLEU-1 最高: {best_name} = {best_bleu1:.4f}")


if __name__ == "__main__":
    main()