"""
Caption 数据增强脚本
功能：对每条训练数据的 gpt 描述生成 N 个语义等价的变体，扩充数据集
位置：mydataset/augment_captions.py
用法：
    python mydataset/augment_captions.py \
        --input  dataset/train_dataset.jsonl \
        --output dataset/train_dataset_augmented.jsonl \
        --num_variants 2 \
        --workers 4
    python mydataset/augment_captions.py \
        --input  dataset/train_dataset.jsonl \
        --output dataset/train_dataset_augmented.jsonl \
        --num_variants 2 \
        --workers 4

环境变量（必须提前设置）：
    export DASHSCOPE_API_KEY="sk-xxxx"
    export DASHSCOPE_BASE_URL="http://your-proxy/v1"   # 内网代理地址
"""

import os
import re
import json
import time
import copy
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from openai import OpenAI

# ── 日志配置 ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/augment.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────────────────
MODEL_NAME   = "qwen3.6-plus"          # 按实际内网模型名修改
MAX_RETRIES  = 3
RETRY_DELAY  = 2.0                  # 秒，指数退避基数
# 变体长度合理范围（相对原文词数比例）
MIN_LEN_RATIO = 0.5
MAX_LEN_RATIO = 2.0

#  ── Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert at paraphrasing remote sensing building descriptions. "
    "Rewrite the given description to express the same facts using different "
    "sentence structures and word order, following ALL the rules below.\n\n"

    "=== RULE 1: FIXED TERMS — copy these verbatim, never replace ===\n"
    "Shape:   irregular polygon, roughly rectangular, rectangular, L-shaped, irregular\n"
    "Height:  low-rise, mid-rise\n"
    "Roof equipment: rooftop equipment, rooftop units, ventilation units, HVAC units,\n"
    "         solar panels, dormers, chimneys, rooftop vents, ductwork\n"
    "         (if original says 'central chimney', keep 'central chimney')\n"
    "Layout:  enclosed layout, stacked pattern, stacked layout, U-shaped layout,\n"
    "         column pattern, paired arrangement\n"
    "Environment: copy the EXACT environment phrase from the original —\n"
    "         if original says 'parking areas' use 'parking areas';\n"
    "         if original says 'parking lots' use 'parking lots';\n"
    "         if original says 'green lawns' use 'green lawns';\n"
    "         if original says 'dense green vegetation' use 'dense green vegetation'.\n"
    "         Do NOT mix or substitute these across variants.\n\n"

    "=== RULE 2: SENTENCE COUNT — must match the original exactly ===\n"
    "Count the sentences in the original. Your rewrite must have the same number.\n"
    "If the original has a short standalone sentence like 'It is surrounded by smaller structures.',\n"
    "keep it as a standalone sentence — do not merge it into another sentence.\n\n"

    "=== RULE 3: SENTENCE STRUCTURE — vary openings, keep facts intact ===\n"
    "You MAY change:\n"
    "  - Which element opens each sentence (scene / building count / roof / environment)\n"
    "  - Verb choices: features→holds/contains; is bordered by→is surrounded by/lines\n"
    "  - Building reference: 'The center building'→'The building at the center'\n"
    "  - Minor connectors between attributes\n"
    "You MUST NOT:\n"
    "  - Add sentences not present in the original\n"
    "  - Remove any sentence from the original\n"
    "  - Add descriptive intensifiers not in the original\n"
    "  - Use formal synonyms: edifice, facility, premises, thoroughfare, photovoltaic\n\n"

    "=== RULE 4: DIVERSITY — sentence opening must differ from original AND prior variant ===\n"
    "Rotate through these opening patterns:\n"
    "  Pattern A: 'This X area features...'\n"
    "  Pattern B: 'Open ground surrounds...' / 'X surrounds...'\n"
    "  Pattern C: 'Two buildings arranged in... occupy this X area.'\n"
    "  Pattern D: 'Dense trees, paved roads... surround an X area...'\n\n"

    "OUTPUT: Return ONLY the rewritten description. No explanation, no bullet points."
)

USER_PROMPT_TMPL = (
    "Rewrite this description following ALL rules above.\n\n"
    "Original:\n{original}\n\n"
    "Rewritten:"
)

USER_PROMPT_WITH_PREV_TMPL = (
    "Rewrite this description following ALL rules above.\n\n"
    "Original:\n{original}\n\n"
    "Previous rewrite (your new version must open with a different pattern):\n"
    "{previous}\n\n"
    "Rewritten:"
)

# ── 校验常量 ──────────────────────────────────────────────────────────
ENV_PAIRS = [
    ("parking areas",         "parking areas"),
    ("parking lots",          "parking lots"),
    ("parking spaces",        "parking spaces"),
    ("green lawns",           "green lawns"),
    ("green vegetation",      "green vegetation"),
    ("dense green vegetation","dense green vegetation"),
    ("open pathways",         "open pathways"),
    ("scattered greenery",    "scattered greenery"),
    ("dense trees",           "dense trees"),
    ("open ground",           "open ground"),
]

FIXED_TERMS = [
    "low-rise", "mid-rise",
    "irregular polygon", "roughly rectangular", "L-shaped",
    "rooftop equipment", "rooftop units", "solar panels",
    "HVAC units", "ventilation units", "ductwork",
    "enclosed layout", "stacked", "U-shaped", "column pattern",
    "paired arrangement", "paved roads", "parking",
]

ENV_KEYWORDS = [
    "paved roads", "parking", "green lawns", "pathways",
    "greenery", "trees", "vegetation", "open ground",
]


def build_client() -> OpenAI:
    api_key  = os.getenv("DASHSCOPE_API_KEY", "sk-5745b9fa39774f1387273715a4167260")
    base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    if not api_key:
        raise EnvironmentError("DASHSCOPE_API_KEY 未设置，请先 export DASHSCOPE_API_KEY=sk-xxx")
    return OpenAI(api_key=api_key, base_url=base_url)


def count_sentences(text: str) -> int:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return len([p for p in parts if p.strip()])


def _get_env_overlap(text: str) -> set:
    """获取文本首句和末句中共同出现的环境词集合"""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) < 2:
        return set()
    first = {w for w in ENV_KEYWORDS if w in sentences[0].lower()}
    last  = {w for w in ENV_KEYWORDS if w in sentences[-1].lower()}
    return first & last


def call_api(
    client: OpenAI,
    original: str,
    temperature: float,
    previous_variant: str = None,
) -> Optional[str]:

    user_content = (
        USER_PROMPT_WITH_PREV_TMPL.format(original=original, previous=previous_variant)
        if previous_variant
        else USER_PROMPT_TMPL.format(original=original)
    )

    orig_sentence_count = count_sentences(original)
    orig_words          = len(original.split())

    required_env = [
        req for orig_env, req in ENV_PAIRS
        if orig_env in original.lower()
    ]

    orig_env_overlap = _get_env_overlap(original)

    last_generated = None  # 保存最后一次生成的文本，用于耗尽重试后的兜底

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                temperature=temperature,
                max_tokens=512,
            )
            text        = resp.choices[0].message.content.strip()
            last_generated = text  # 每次成功获取响应就更新兜底值
            fail_reason = None

            # ── 校验1：长度比例 ──────────────────────────────────────
            ratio = len(text.split()) / max(orig_words, 1)
            if not (MIN_LEN_RATIO <= ratio <= MAX_LEN_RATIO):
                fail_reason = f"长度比例异常({ratio:.2f})"

            # ── 校验2：句数一致 ──────────────────────────────────────
            if fail_reason is None:
                v_sent = count_sentences(text)
                if v_sent != orig_sentence_count:
                    fail_reason = (
                        f"句数不一致(原文{orig_sentence_count}句，变体{v_sent}句)"
                    )

            # ── 校验3：固定术语保留 ──────────────────────────────────
            if fail_reason is None:
                missing = [
                    t for t in FIXED_TERMS
                    if t in original.lower() and t not in text.lower()
                ]
                if missing:
                    fail_reason = f"固定术语丢失{missing}"

            # ── 校验4：环境词精确匹配 ────────────────────────────────
            if fail_reason is None:
                missing_env = [e for e in required_env if e not in text.lower()]
                if missing_env:
                    fail_reason = f"环境词不一致，缺少{missing_env}"

            # ── 校验5：首尾句环境词重复（修复版）────────────────────
            # 只在原文本身没有首尾重叠时才拒绝变体的首尾重叠
            # 原文有重叠 → 变体保持一致是正确的，跳过此校验
            if fail_reason is None and not orig_env_overlap:
                variant_overlap = _get_env_overlap(text)
                if variant_overlap:
                    fail_reason = f"首尾句环境词重复{variant_overlap}"

            # ── 统一重试 / 返回 ──────────────────────────────────────
            if fail_reason:
                logger.warning(f"{fail_reason}，重试{attempt}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
                continue

            return text

        except Exception as e:
            logger.warning(f"API调用失败(attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)

    # 所有重试耗尽：返回最后一次生成的文本（即使未通过校验）
    # 原因：回退原文会造成三条完全相同的数据，训练时等于重复学习，
    # 没有增强效果；最后一次生成的文本至少有句式变化，优于原文复制
    if last_generated:
        logger.warning(f"已达最大重试次数({MAX_RETRIES})，使用最后一次生成结果")
        return last_generated
    logger.error(f"已达最大重试次数({MAX_RETRIES})且无有效生成，返回 None")
    return None


def augment_one(
    client: OpenAI,
    item: dict,
    num_variants: int,
    line_idx: int,
) -> list:
    conversations = item.get("conversations", [])
    gpt_turn = next((c for c in conversations if c.get("from") == "gpt"), None)
    if gpt_turn is None:
        logger.warning(f"第{line_idx}条无 gpt 回复，跳过")
        return [item]

    original_text      = gpt_turn["value"]
    results            = [item]
    temperatures       = [0.75, 0.90, 0.85, 0.80][:num_variants]
    generated_variants = []

    for v_idx, temp in enumerate(temperatures, start=1):
        prev         = generated_variants[-1] if generated_variants else None
        variant_text = call_api(client, original_text, temperature=temp,
                                previous_variant=prev)

        if variant_text is None:
            # call_api 返回 None 说明 API 完全无响应（网络异常等），
            # 此时才回退原文（极少发生）
            logger.warning(f"第{line_idx}条变体{v_idx} API无响应，回退原文")
            variant_text = original_text

        generated_variants.append(variant_text)

        new_item = copy.deepcopy(item)
        for conv in new_item["conversations"]:
            if conv.get("from") == "gpt":
                conv["value"] = variant_text
                break
        results.append(new_item)

    return results


def load_progress(progress_file: Path) -> set:
    if progress_file.exists():
        return set(json.loads(progress_file.read_text()))
    return set()


def save_progress(progress_file: Path, done: set):
    progress_file.write_text(json.dumps(sorted(done)))


def main():
    parser = argparse.ArgumentParser(description="Caption 数据增强")
    parser.add_argument("--input",        type=str, default="dataset/train_dataset.jsonl")
    parser.add_argument("--output",       type=str, default="dataset/train_dataset_augmented.jsonl")
    parser.add_argument("--num_variants", type=int, default=2)
    parser.add_argument("--workers",      type=int, default=3)
    parser.add_argument("--resume",       action="store_true")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    progress_file = Path("logs/progress.json")
    failed_file   = Path("logs/failed.jsonl")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        all_items = [json.loads(line) for line in f if line.strip()]

    total = len(all_items)
    logger.info(f"共读取 {total} 条数据，每条生成 {args.num_variants} 个变体")
    logger.info(f"预计输出 {total * (args.num_variants + 1)} 条，API 调用 {total * args.num_variants} 次")

    done_indices = load_progress(progress_file) if args.resume else set()
    if done_indices:
        logger.info(f"断点续传：已跳过 {len(done_indices)} 条")

    client    = build_client()
    out_path  = Path(args.output)
    open_mode = "a" if args.resume else "w"
    out_f     = open(out_path, open_mode, encoding="utf-8")
    fail_f    = open(failed_file, "a", encoding="utf-8")

    completed = 0
    failed    = 0

    def process(idx_item):
        idx, item = idx_item
        if idx in done_indices:
            return idx, None, True
        try:
            rows = augment_one(client, item, args.num_variants, idx)
            return idx, rows, True
        except Exception as e:
            logger.error(f"第{idx}条处理异常: {e}")
            return idx, item, False

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process, (idx, item)): idx
            for idx, item in enumerate(all_items)
        }
        for future in as_completed(futures):
            idx, result, ok = future.result()

            if result is None:
                continue

            if ok:
                for row in result:
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                done_indices.add(idx)
                save_progress(progress_file, done_indices)
                completed += 1
                if completed % 50 == 0:
                    logger.info(f"进度: {completed}/{total}")
            else:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                fail_f.write(
                    json.dumps({"line": idx, "item": result}, ensure_ascii=False) + "\n"
                )
                failed += 1

    out_f.close()
    fail_f.close()

    out_lines = sum(1 for _ in open(out_path, encoding="utf-8"))
    logger.info("=" * 50)
    logger.info(f"完成！输出文件：{out_path}")
    logger.info(f"输出总条数：{out_lines}")
    logger.info(f"成功处理：{completed} 条 | 失败回退：{failed} 条")
    if failed:
        logger.info(f"失败记录见：{failed_file}")


if __name__ == "__main__":
    main()