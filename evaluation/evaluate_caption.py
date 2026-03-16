import json
import logging
import os
import argparse
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from evaluate import load
from sentence_transformers import SentenceTransformer
from bert_score import score as bert_score
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CaptionEvaluator:
    """通用的 Caption 评估器"""

    def __init__(
        self,
        t5_model_path: Optional[str] = None,
        bert_model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化评估器

        Args:
            t5_model_path: Sentence-T5 模型路径，默认使用 sentence-transformers/sentence-t5-base
            bert_model_path: BERT 模型路径，默认使用 roberta-large
            device: 设备
        """
        self.device = device

        if t5_model_path:
            self.t5_model = SentenceTransformer(t5_model_path, device=device)
        else:
            self.t5_model = SentenceTransformer("sentence-transformers/sentence-t5-base", device=device)

        if bert_model_path:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
            self.bert_model = AutoModel.from_pretrained(bert_model_path).to(device)
        else:
            self.bert_tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")
            self.bert_model = AutoModel.from_pretrained("FacebookAI/roberta-large").to(device)

        self.bleu = load("bleu")
        self.rouge = load("rouge")
        self.meteor = load("meteor")

    def calculate_bleu(self, candidates: List[str], references: List[List[str]]) -> Dict:
        """计算 BLEU 分数"""
        results = self.bleu.compute(
            predictions=candidates,
            references=references
        )
        return {
            "bleu": float(results["bleu"]),
            "bleu1": float(results["precisions"][0]) if results["precisions"] else 0.0,
            "bleu2": float(results["precisions"][1]) if len(results["precisions"]) > 1 else 0.0,
            "bleu3": float(results["precisions"][2]) if len(results["precisions"]) > 2 else 0.0,
            "bleu4": float(results["precisions"][3]) if len(results["precisions"]) > 3 else 0.0,
        }

    def calculate_rouge(self, candidates: List[str], references: List[str]) -> Dict:
        """计算 ROUGE 分数"""
        results = self.rouge.compute(
            predictions=candidates,
            references=references
        )
        return {
            "rouge1": float(results["rouge1"]),
            "rouge2": float(results["rouge2"]),
            "rougeL": float(results["rougeL"]),
            "rougeLsum": float(results["rougeLsum"]),
        }

    def calculate_meteor(self, candidates: List[str], references: List[str]) -> Dict:
        """计算 METEOR 分数"""
        results = self.meteor.compute(
            predictions=candidates,
            references=references
        )
        return {"meteor": float(results["meteor"])}

    def calculate_bertscore(self, candidates: List[str], references: List[str]) -> Dict:
        """计算 BERTScore"""
        P, R, F1 = bert_score(
            candidates,
            references,
            model_type="FacebookAI/roberta-large",
            lang="en",
            verbose=False
        )
        return {
            "bertscore_p": float(P.mean().item()),
            "bertscore_r": float(R.mean().item()),
            "bertscore_f1": float(F1.mean().item()),
        }

    def calculate_t5_similarity(self, candidates: List[str], references: List[str]) -> Dict:
        """计算 Sentence-T5 余弦相似度"""
        candidate_embeds = self.t5_model.encode(candidates, normalize_embeddings=True)
        reference_embeds = self.t5_model.encode(references, normalize_embeddings=True)
        similarities = cosine_similarity(candidate_embeds, reference_embeds).diagonal()
        cubed_similarities = np.abs(similarities ** 3)
        return {
            "t5_cosine_similarity": float(similarities.mean()),
            "t5_cubed_similarity": float(cubed_similarities.mean()),
        }

    def calculate_caption_length(self, candidates: List[str]) -> Dict:
        """计算 Caption 长度统计"""
        lengths = [len(caption.split()) for caption in candidates]
        return {
            "avg_word_count": float(np.mean(lengths)),
            "std_word_count": float(np.std(lengths)),
            "min_word_count": int(np.min(lengths)),
            "max_word_count": int(np.max(lengths)),
        }

    def evaluate(
        self,
        candidates: List[str],
        references: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict:
        """
        评估生成的 Caption

        Args:
            candidates: 生成的 Caption 列表
            references: 参考 Caption 列表（每个元素是单个参考或参考列表）
            metrics: 要计算的指标列表，默认计算所有指标

        Returns:
            包含所有指标的字典
        """
        if len(candidates) != len(references):
            raise ValueError(f"Candidates count ({len(candidates)}) != references count ({len(references)})")

        if not metrics:
            metrics = ["bleu", "rouge", "meteor", "bertscore", "t5_similarity", "length"]

        results = {}

        references_for_bleu = []
        for ref in references:
            if isinstance(ref, str):
                references_for_bleu.append([ref])
            else:
                references_for_bleu.append(ref)

        references_flat = []
        for ref in references:
            if isinstance(ref, str):
                references_flat.append(ref)
            else:
                references_flat.append(ref[0])

        if "bleu" in metrics:
            logger.info("Calculating BLEU...")
            results.update(self.calculate_bleu(candidates, references_for_bleu))

        if "rouge" in metrics:
            logger.info("Calculating ROUGE...")
            results.update(self.calculate_rouge(candidates, references_flat))

        if "meteor" in metrics:
            logger.info("Calculating METEOR...")
            results.update(self.calculate_meteor(candidates, references_flat))

        if "bertscore" in metrics:
            logger.info("Calculating BERTScore...")
            results.update(self.calculate_bertscore(candidates, references_flat))

        if "t5_similarity" in metrics:
            logger.info("Calculating T5 Similarity...")
            results.update(self.calculate_t5_similarity(candidates, references_flat))

        if "length" in metrics:
            logger.info("Calculating Caption Length...")
            results.update(self.calculate_caption_length(candidates))

        return results


def load_jsonl(file_path: str) -> List[Dict]:
    """加载 JSONL 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def load_json(file_path: str) -> List[Dict]:
    """加载 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="通用 Caption 评估工具")
    parser.add_argument(
        "--prediction_file",
        type=str,
        required=True,
        help="预测结果文件路径 (JSON 或 JSONL)"
    )
    parser.add_argument(
        "--reference_file",
        type=str,
        required=True,
        help="参考结果文件路径 (JSON 或 JSONL)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results.json",
        help="评估结果输出文件"
    )
    parser.add_argument(
        "--prediction_key",
        type=str,
        default="prediction",
        help="预测结果在 JSON 中的键名"
    )
    parser.add_argument(
        "--reference_key",
        type=str,
        default="reference",
        help="参考结果在 JSON 中的键名"
    )
    parser.add_argument(
        "--id_key",
        type=str,
        default="id",
        help="样本 ID 键名，用于匹配预测和参考"
    )
    parser.add_argument(
        "--t5_model_path",
        type=str,
        default=None,
        help="Sentence-T5 模型路径"
    )
    parser.add_argument(
        "--bert_model_path",
        type=str,
        default=None,
        help="BERT 模型路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Caption Evaluation Tool")
    logger.info("=" * 60)

    logger.info(f"Loading predictions from: {args.prediction_file}")
    if args.prediction_file.endswith(".jsonl"):
        predictions = load_jsonl(args.prediction_file)
    else:
        predictions = load_json(args.prediction_file)

    logger.info(f"Loading references from: {args.reference_file}")
    if args.reference_file.endswith(".jsonl"):
        references = load_jsonl(args.reference_file)
    else:
        references = load_json(args.reference_file)

    logger.info(f"Loaded {len(predictions)} predictions and {len(references)} references")

    ref_dict = {}
    for ref in references:
        ref_dict[ref[args.id_key]] = ref[args.reference_key]

    candidate_list = []
    reference_list = []
    matched_count = 0

    for pred in predictions:
        sample_id = pred[args.id_key]
        if sample_id in ref_dict:
            candidate_list.append(pred[args.prediction_key])
            reference_list.append(ref_dict[sample_id])
            matched_count += 1

    logger.info(f"Matched {matched_count} samples")

    if matched_count == 0:
        logger.error("No samples matched! Please check your data format.")
        return

    logger.info("Initializing evaluator...")
    evaluator = CaptionEvaluator(
        t5_model_path=args.t5_model_path,
        bert_model_path=args.bert_model_path,
        device=args.device
    )

    logger.info("Running evaluation...")
    results = evaluator.evaluate(candidate_list, reference_list)

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Results:")
    logger.info("=" * 60)
    for key, value in results.items():
        logger.info(f"{key:25s}: {value:.4f}" if isinstance(value, float) else f"{key:25s}: {value}")

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
