import json
import logging
import os
import argparse
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
# --- 修改点 1: 移除 evaluate.load，改用本地库 ---
# from evaluate import load  <-- 删掉或注释掉
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
# ----------------------------------------------
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
    """通用的 Caption 评估器 (已修改为离线稳定版)"""

    def __init__(
        self,
        t5_model_path: Optional[str] = None,
        bert_model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
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

        # --- 修改点 2: 使用本地 rouge_scorer 初始化，移除需要联网的 load ---
        self.rouge_evaluator = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
        # self.bleu = load("bleu")   <-- 移除
        # self.rouge = load("rouge") <-- 移除
        # self.meteor = None         <-- 离线环境建议关掉 METEOR
        # -----------------------------------------------------------------

    def calculate_bleu(self, candidates: List[str], references: List[List[str]]) -> Dict:
        """修改点 3: 使用 NLTK 本地计算 BLEU (无需联网脚本)"""
        # 分词处理
        hypotheses = [c.split() for c in candidates]
        # reference 格式：[[[ref1_tokens], [ref2_tokens]], ...]
        list_of_references = [[r.split() for r in refs] for refs in references]
        
        # 使用平滑函数防止短句得分出现 0
        chencherry = SmoothingFunction()
        
        # 计算各阶 BLEU
        b1 = corpus_bleu(list_of_references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=chencherry.method1)
        b2 = corpus_bleu(list_of_references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1)
        b3 = corpus_bleu(list_of_references, hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=chencherry.method1)
        b4 = corpus_bleu(list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
        
        return {
            "bleu": float(b4),
            "bleu1": float(b1),
            "bleu2": float(b2),
            "bleu3": float(b3),
            "bleu4": float(b4),
        }

    def calculate_rouge(self, candidates: List[str], references: List[str]) -> Dict:
        """修改点 4: 使用 rouge-score 本地计算 (无需联网脚本)"""
        all_scores = []
        for cand, ref in zip(candidates, references):
            score = self.rouge_evaluator.score(ref, cand)
            all_scores.append(score)
            
        return {
            "rouge1": float(np.mean([s['rouge1'].fmeasure for s in all_scores])),
            "rouge2": float(np.mean([s['rouge2'].fmeasure for s in all_scores])),
            "rougeL": float(np.mean([s['rougeL'].fmeasure for s in all_scores])),
            "rougeLsum": float(np.mean([s['rougeLsum'].fmeasure for s in all_scores])),
        }

    def calculate_meteor(self, candidates: List[str], references: List[str]) -> Dict:
        """离线环境下跳过 METEOR"""
        return {"meteor": 0.0}

    def calculate_bertscore(self, candidates: List[str], references: List[str]) -> Dict:
        """计算 BERTScore (已修复本地路径导致的 KeyError)"""
        # 获取本地模型路径
        model_path = self.bert_model.config._name_or_path
        
        P, R, F1 = bert_score(
            candidates,
            references,
            model_type=model_path, 
            num_layers=17, # <--- 关键修改：手动指定 RoBERTa-Large 的推荐层数（17）
            lang="en",
            device=self.device,
            verbose=False
        )
        return {
            "bertscore_p": float(P.mean().item()),
            "bertscore_r": float(R.mean().item()),
            "bertscore_f1": float(F1.mean().item()),
        }

    def calculate_t5_similarity(self, candidates: List[str], references: List[str]) -> Dict:
        """计算 Sentence-T5 余弦相似度 (已指向本地模型)"""
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
        if len(candidates) != len(references):
            raise ValueError(f"Candidates count ({len(candidates)}) != references count ({len(references)})")

        # --- 修改点 5: 默认指标移除 meteor ---
        if not metrics:
            metrics = ["bleu", "rouge", "bertscore", "t5_similarity", "length"]
        # ----------------------------------

        results = {}

        # 处理参考答案格式
        references_for_bleu = [[ref] for ref in references]
        references_flat = references

        if "bleu" in metrics:
            logger.info("Calculating BLEU (NLTK Offline)...")
            results.update(self.calculate_bleu(candidates, references_for_bleu))

        if "rouge" in metrics:
            logger.info("Calculating ROUGE (Rouge-Score Offline)...")
            results.update(self.calculate_rouge(candidates, references_flat))

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
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def load_json(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="通用 Caption 评估工具 (本地版)")
    parser.add_argument("--prediction_file", type=str, required=True)
    parser.add_argument("--reference_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="evaluation_results.json")
    parser.add_argument("--prediction_key", type=str, default="prediction")
    parser.add_argument("--reference_key", type=str, default="conversations") # 默认设为 conversations
    parser.add_argument("--id_key", type=str, default="id")
    parser.add_argument("--t5_model_path", type=str, default=None)
    parser.add_argument("--bert_model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Caption Evaluation Tool (Offline Stable)")
    logger.info("=" * 60)

    if args.prediction_file.endswith(".jsonl"):
        predictions = load_jsonl(args.prediction_file)
    else:
        predictions = load_json(args.prediction_file)

    if args.reference_file.endswith(".jsonl"):
        references = load_jsonl(args.reference_file)
    else:
        references = load_json(args.reference_file)

    # --- 修改点 6: 修正参考答案的解析逻辑 (处理 conversations 列表) ---
    ref_dict = {}
    for ref in references:
        val = ref[args.reference_key]
        if isinstance(val, list):
            # 找到 gpt 的回复
            caption = next((m["value"] for m in val if m.get("from") == "gpt"), "")
            ref_dict[ref[args.id_key]] = caption
        else:
            ref_dict[ref[args.id_key]] = str(val)
    # -------------------------------------------------------------

    candidate_list = []
    reference_list = []
    for pred in predictions:
        sample_id = pred[args.id_key]
        if sample_id in ref_dict:
            candidate_list.append(pred[args.prediction_key])
            reference_list.append(ref_dict[sample_id])

    logger.info(f"Matched {len(candidate_list)} samples")

    evaluator = CaptionEvaluator(
        t5_model_path=args.t5_model_path,
        bert_model_path=args.bert_model_path,
        device=args.device
    )

    results = evaluator.evaluate(candidate_list, reference_list)

    # 输出结果
    for key, value in results.items():
        logger.info(f"{key:25s}: {value:.4f}" if isinstance(value, float) else f"{key:25s}: {value}")

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()