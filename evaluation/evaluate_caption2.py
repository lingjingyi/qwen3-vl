"""
Caption 评估脚本（LAVIS / RSGPT 对齐版）
指标：BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr, SPICE

与原版的核心差异
─────────────────────────────────────────────────────────────────────
  原版：BLEU/METEOR 用 Python .split() 分词 + nltk 实现
  本版：所有指标统一先经 PTBTokenizer（Stanford CoreNLP）分词，
        再调用 pycocoevalcap 内置的 Bleu/Meteor/Rouge/Cider/Spice。
        与 RSGPT、COCO caption 评测标准完全一致，数字可直接比较。

依赖
─────────────────────────────────────────────────────────────────────
  pip install pycocoevalcap pycocotools
  # PTBTokenizer / METEOR / SPICE 需要 Java（JRE 8+）
"""

import json
import logging
import argparse
from typing import Dict, List
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── pycocoevalcap 全家桶 ──────────────────────────────────────────────
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu             import Bleu
from pycocoevalcap.meteor.meteor         import Meteor
from pycocoevalcap.rouge.rouge           import Rouge
from pycocoevalcap.cider.cider           import Cider
from pycocoevalcap.spice.spice           import Spice


# ── 自定义 JSON encoder，保留小数点后四位 ─────────────────────────────
class RoundedFloatEncoder(json.JSONEncoder):
    DECIMAL_PLACES = 4

    def iterencode(self, obj, _one_shot=False):
        return super().iterencode(self._round(obj), _one_shot)

    def _round(self, obj):
        if isinstance(obj, float):
            return round(obj, self.DECIMAL_PLACES)
        if isinstance(obj, dict):
            return {k: self._round(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._round(v) for v in obj]
        return obj


# ══════════════════════════════════════════════════════════════════════
# 核心评估类
# ══════════════════════════════════════════════════════════════════════

class CaptionEvaluator:
    """
    LAVIS / RSGPT 风格的 Caption 评估器。

    输入格式与原版相同（candidates/references 列表），
    内部统一通过 PTBTokenizer 分词后再送入各 scorer，
    保证与 COCO caption 社区的评测标准完全一致。
    """

    # ── PTBTokenizer 期望的输入格式转换 ──────────────────────────────
    @staticmethod
    def _to_ptb_fmt(
        candidates: List[str],
        references: List[List[str]],
    ):
        """
        将列表格式转换为 PTBTokenizer 所需的 dict 格式：
          gts: {idx: [{"caption": ref1}, {"caption": ref2}, ...]}
          res: {idx: [{"caption": pred}]}
        """
        gts = {
            i: [{"caption": r} for r in refs]
            for i, refs in enumerate(references)
        }
        res = {
            i: [{"caption": candidates[i]}]
            for i in range(len(candidates))
        }
        return gts, res

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

        gts, res = self._to_ptb_fmt(candidates, references)

        # ── Step 1: PTBTokenizer 统一分词 ─────────────────────────────
        # 这是与原版最关键的差异：所有指标共享同一份标准化 token 序列，
        # 与 RSGPT / COCO caption 已发表结果保持一致。
        logger.info("Tokenizing with PTBTokenizer (requires Java)...")
        tokenizer = PTBTokenizer()
        gts_tok   = tokenizer.tokenize(gts)
        res_tok   = tokenizer.tokenize(res)

        results = {}

        # ── Step 2: BLEU-1/2/3/4 ─────────────────────────────────────
        logger.info("Calculating BLEU-1/2/3/4...")
        bleu_scorer = Bleu(4)
        bleu_scores, _ = bleu_scorer.compute_score(gts_tok, res_tok)
        # bleu_scores 是长度为 4 的列表：[bleu1, bleu2, bleu3, bleu4]
        for n, score in enumerate(bleu_scores, start=1):
            results[f"bleu{n}"] = float(score)

        # ── Step 3: METEOR ────────────────────────────────────────────
        logger.info("Calculating METEOR (requires Java)...")
        meteor_scorer = Meteor()
        meteor_score, _ = meteor_scorer.compute_score(gts_tok, res_tok)
        results["meteor"] = float(meteor_score)

        # ── Step 4: ROUGE-L ───────────────────────────────────────────
        logger.info("Calculating ROUGE-L...")
        rouge_scorer = Rouge()
        rouge_score, _ = rouge_scorer.compute_score(gts_tok, res_tok)
        results["rougeL"] = float(rouge_score)

        # ── Step 5: CIDEr ─────────────────────────────────────────────
        logger.info("Calculating CIDEr...")
        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(gts_tok, res_tok)
        results["cider"] = float(cider_score)

        # ── Step 6: SPICE ─────────────────────────────────────────────
        if not skip_spice:
            logger.info("Calculating SPICE (requires Java)...")
            try:
                spice_scorer = Spice()
                spice_score, _ = spice_scorer.compute_score(gts_tok, res_tok)
                results["spice"] = float(spice_score)
            except Exception as e:
                logger.warning(f"SPICE 计算失败（需要 Java 环境）: {e}")
                results["spice"] = -1.0

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
        description=(
            "Caption 评估（LAVIS/RSGPT 对齐版）\n"
            "所有指标统一经 PTBTokenizer 分词，与 COCO caption 社区标准一致。\n"
            "指标：BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr, SPICE"
        )
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

    # ── 保存结果 ──────────────────────────────────────────────────────
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=RoundedFloatEncoder)
    logger.info(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()