"""
Caption 评估脚本
指标：BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr, SPICE
"""
import json
import logging
import argparse
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

for pkg in ['punkt', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'tokenizers/{pkg}' if pkg == 'punkt' else f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg)


def _to_coco_fmt(candidates: List[str], references: List[List[str]]):
    gts = {i: refs            for i, refs in enumerate(references)}
    res = {i: [candidates[i]] for i in range(len(candidates))}
    return gts, res


# ── 修复：自定义 JSON encoder，保留小数点后四位 ────────────────────────
class RoundedFloatEncoder(json.JSONEncoder):
    """
    将所有 float 值四舍五入到小数点后四位再序列化。
    json.dump 默认输出完整精度（如 0.7450123456789012），
    此 encoder 保证输出为 0.7450，与控制台打印格式一致。
    """
    DECIMAL_PLACES = 4

    def iterencode(self, obj, _one_shot=False):
        # 递归处理，保证嵌套结构中的 float 也被处理
        return super().iterencode(self._round(obj), _one_shot)

    def _round(self, obj):
        if isinstance(obj, float):
            return round(obj, self.DECIMAL_PLACES)
        if isinstance(obj, dict):
            return {k: self._round(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._round(v) for v in obj]
        return obj


class CaptionEvaluator:

    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True
        )
        self.cider_scorer = Cider()
        self.spice_scorer = Spice()

    # ── BLEU-1/2/3/4 ─────────────────────────────────────────────────
    def calculate_bleu(
        self,
        candidates: List[str],
        references: List[List[str]],
    ) -> Dict:
        hyps = [c.split() for c in candidates]
        refs = [[r.split() for r in rs] for rs in references]
        sf   = SmoothingFunction()
        return {
            "bleu1": float(corpus_bleu(refs, hyps, weights=(1, 0, 0, 0),          smoothing_function=sf.method1)),
            "bleu2": float(corpus_bleu(refs, hyps, weights=(.5, .5, 0, 0),        smoothing_function=sf.method1)),
            "bleu3": float(corpus_bleu(refs, hyps, weights=(.33, .33, .33, 0),    smoothing_function=sf.method1)),
            "bleu4": float(corpus_bleu(refs, hyps, weights=(.25, .25, .25, .25),  smoothing_function=sf.method1)),
        }

    # ── METEOR ────────────────────────────────────────────────────────
    def calculate_meteor(
        self,
        candidates: List[str],
        references: List[List[str]],
    ) -> Dict:
        scores = [
            meteor_score([r.split() for r in refs], cand.split())
            for cand, refs in zip(candidates, references)
        ]
        return {"meteor": float(np.mean(scores))}

    # ── ROUGE-L ───────────────────────────────────────────────────────
    def calculate_rouge(
        self,
        candidates: List[str],
        references: List[List[str]],
    ) -> Dict:
        rL = []
        for cand, refs in zip(candidates, references):
            best = max(self.rouge.score(ref, cand)['rougeL'].fmeasure for ref in refs)
            rL.append(best)
        return {"rougeL": float(np.mean(rL))}

    # ── CIDEr ─────────────────────────────────────────────────────────
    def calculate_cider(
        self,
        candidates: List[str],
        references: List[List[str]],
    ) -> Dict:
        gts, res = _to_coco_fmt(candidates, references)
        score, _ = self.cider_scorer.compute_score(gts, res)
        return {"cider": float(score)}

    # ── SPICE ─────────────────────────────────────────────────────────
    def calculate_spice(
        self,
        candidates: List[str],
        references: List[List[str]],
    ) -> Dict:
        gts, res = _to_coco_fmt(candidates, references)
        try:
            score, _ = self.spice_scorer.compute_score(gts, res)
            return {"spice": float(score)}
        except Exception as e:
            logger.warning(f"SPICE 计算失败（需要 Java 环境）: {e}")
            return {"spice": -1.0}

    # ── 主评估入口 ────────────────────────────────────────────────────
    def evaluate(
        self,
        candidates: List[str],
        references: List[List[str]],
        skip_spice: bool = False,
    ) -> Dict:
        if len(candidates) != len(references):
            raise ValueError(
                f"candidates({len(candidates)}) != references({len(references)})"
            )
        results = {}

        logger.info("Calculating BLEU-1/2/3/4...")
        results.update(self.calculate_bleu(candidates, references))

        logger.info("Calculating METEOR...")
        results.update(self.calculate_meteor(candidates, references))

        logger.info("Calculating ROUGE-L...")
        results.update(self.calculate_rouge(candidates, references))

        logger.info("Calculating CIDEr...")
        results.update(self.calculate_cider(candidates, references))

        if not skip_spice:
            logger.info("Calculating SPICE (requires Java)...")
            results.update(self.calculate_spice(candidates, references))

        return results


# ── 结果打印 ──────────────────────────────────────────────────────────
METRIC_DISPLAY_ORDER = ["bleu1", "bleu2", "bleu3", "bleu4", "meteor", "rougeL", "cider", "spice"]

METRIC_LABELS = {
    "bleu1":  "BLEU-1 ",
    "bleu2":  "BLEU-2 ",
    "bleu3":  "BLEU-3 ",
    "bleu4":  "BLEU-4 ",
    "meteor": "METEOR ",
    "rougeL": "ROUGE-L",
    "cider":  "CIDEr  ",
    "spice":  "SPICE  ",
}


def print_results(results: Dict) -> None:
    sep = "=" * 35
    logger.info("\n" + sep)
    logger.info("  Evaluation Results")
    logger.info(sep)
    for k in METRIC_DISPLAY_ORDER:
        if k not in results:
            continue
        v     = results[k]
        label = METRIC_LABELS.get(k, k)
        if v < 0:
            logger.info(f"  {label} : NA")
        else:
            logger.info(f"  {label} : {v:.4f}")
    logger.info(sep)


# ── 文件加载 ──────────────────────────────────────────────────────────
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def extract_gpt_caption(item: dict, reference_key: str) -> str:
    val = item.get(reference_key, "")
    if isinstance(val, list):
        return next((m["value"] for m in val if m.get("from") == "gpt"), "")
    return str(val)


# ── CLI ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Caption 评估 — BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr, SPICE"
    )
    parser.add_argument("--prediction_file", type=str, required=True)
    parser.add_argument("--reference_file",  type=str, default=None)
    parser.add_argument("--output_file",     type=str, default="evaluation_results.json")
    parser.add_argument("--prediction_key",  type=str, default="prediction")
    parser.add_argument("--reference_key",   type=str, default="conversations")
    parser.add_argument("--id_key",          type=str, default="id")
    parser.add_argument("--skip_spice",      action="store_true",
                        help="跳过 SPICE（无 Java 环境时使用）")
    args = parser.parse_args()

    preds = (load_jsonl if args.prediction_file.endswith(".jsonl")
             else load_json)(args.prediction_file)

    # ── 构建参考答案字典 ──────────────────────────────────────────────
    ref_dict: Dict[str, List[str]] = defaultdict(list)

    if args.reference_file is None:
        logger.info("单文件模式：从 prediction_file 的 conversations 字段读取参考答案")
        for item in preds:
            caption = extract_gpt_caption(item, args.reference_key)
            if caption:
                ref_dict[item[args.id_key]].append(caption)
    else:
        logger.info(f"双文件模式：从 {args.reference_file} 读取参考答案")
        refs = (load_jsonl if args.reference_file.endswith(".jsonl")
                else load_json)(args.reference_file)
        for ref in refs:
            caption = extract_gpt_caption(ref, args.reference_key)
            if caption:
                ref_dict[ref[args.id_key]].append(caption)

    # ── 对齐预测与参考 ────────────────────────────────────────────────
    candidate_list, reference_list = [], []
    for pred in preds:
        sid = pred[args.id_key]
        if sid in ref_dict:
            candidate_list.append(pred[args.prediction_key])
            reference_list.append(ref_dict[sid])

    logger.info(f"Matched {len(candidate_list)} samples")

    # ── 评估 ──────────────────────────────────────────────────────────
    evaluator = CaptionEvaluator()
    results   = evaluator.evaluate(
        candidate_list,
        reference_list,
        skip_spice=args.skip_spice,
    )

    # ── 打印结果 ──────────────────────────────────────────────────────
    print_results(results)

    # ── 保存结果：使用 RoundedFloatEncoder 保证四位小数 ──────────────
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=RoundedFloatEncoder)
    logger.info(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()