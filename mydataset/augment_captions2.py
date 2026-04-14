"""
Caption 数据增强脚本（Token 优化版 V2）
功能：对每条训练数据的 gpt 描述生成 N 个语义等价的变体，扩充数据集

【输入格式】
  {"images": [...], "conversations": [{"from": "human", ...}, {"from": "gpt", ...}]}

【输出格式】
  与输入完全一致，不新增任何字段：
    原始条目:  {"images": [...], "conversations": [...原始 gpt ...]}
    变体条目:  {"images": [...], "conversations": [...变体 gpt ...]}

【Token 优化项】
  1. 合并生成：单次请求返回 2 个变体，System Prompt 开销减半
  2. System Prompt 精简 15%：删冗余措辞，规则内容完整保留
  3. previous 无需传入：合并调用后 Variant 1/2 天然要求不同开头
  4. max_tokens 动态计算：按原文长度 ×2.5×2，避免固定 512 浪费
  5. 重试次数 3→2：重试耗尽后使用最后一次生成结果（不回退原文）

【650条预计消耗】~483,000 token（原版超限方案 ~1,041,267 token）
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
MODEL_NAME       = "qwen3.5-plus"
MAX_RETRIES      = 2
RETRY_DELAY      = 2.0
MIN_LEN_RATIO    = 0.5
MAX_LEN_RATIO    = 2.0
MAX_TOKENS_RATIO = 2.5      # 动态 max_tokens：原文词数 × 2.5 × 2个变体
MAX_TOKENS_FLOOR = 200
MAX_TOKENS_CAP   = 600

# ── System Prompt（精简 15%，四条规则和所有 FIXED TERMS 完整保留）────
SYSTEM_PROMPT = (
    "You are an expert at paraphrasing remote sensing building descriptions. "
    "Rewrite the given description using different sentence structures, "
    "following ALL rules below.\n\n"

    "=== RULE 1: FIXED TERMS — copy verbatim, never replace ===\n"
    "Shape: irregular polygon, roughly rectangular, rectangular, L-shaped, irregular\n"
    "Height: low-rise, mid-rise\n"
    "Roof: rooftop equipment, rooftop units, ventilation units, HVAC units, "
    "solar panels, dormers, chimneys, rooftop vents, ductwork\n"
    "Layout: enclosed layout, stacked pattern, stacked layout, U-shaped layout, "
    "column pattern, paired arrangement\n"
    "Environment: copy the EXACT phrase from the original — "
    "'parking areas', 'parking lots', 'green lawns', 'dense green vegetation' "
    "are NOT interchangeable. Do NOT mix or substitute.\n\n"

    "=== RULE 2: SENTENCE COUNT — must match the original exactly ===\n"
    "If the original has a short standalone sentence like "
    "'It is surrounded by smaller structures.', keep it standalone.\n\n"

    "=== RULE 3: SENTENCE STRUCTURE — vary openings, keep all facts ===\n"
    "Change: sentence openings, verb choices (features→holds/contains), "
    "minor connectors. "
    "Do NOT: add/remove sentences, add intensifiers, "
    "use edifice/facility/premises/thoroughfare/photovoltaic.\n\n"

    "=== RULE 4: DIVERSITY — each variant must open differently ===\n"
    "Rotate: Pattern A 'This X area features...' | "
    "Pattern B 'X surrounds...' | "
    "Pattern C 'Two buildings... occupy this X area.' | "
    "Pattern D 'Dense trees, paved roads... surround an X area...'\n\n"

    "OUTPUT FORMAT:\n"
    "Variant 1: <rewrite using one opening pattern>\n"
    "Variant 2: <rewrite using a DIFFERENT opening pattern>\n"
    "No explanation, no extra text."
)

USER_PROMPT_TMPL = (
    "Original:\n{original}\n\n"
    "Generate 2 variants with different sentence opening patterns."
)

# ── 校验常量（与原版完全一致）─────────────────────────────────────────
ENV_PAIRS = [
    ("parking areas",          "parking areas"),
    ("parking lots",           "parking lots"),
    ("parking spaces",         "parking spaces"),
    ("green lawns",            "green lawns"),
    ("green vegetation",       "green vegetation"),
    ("dense green vegetation", "dense green vegetation"),
    ("open pathways",          "open pathways"),
    ("scattered greenery",     "scattered greenery"),
    ("dense trees",            "dense trees"),
    ("open ground",            "open ground"),
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


# ── 工具函数 ──────────────────────────────────────────────────────────

def build_client() -> OpenAI:
    api_key  = os.getenv("DASHSCOPE_API_KEY", "sk-5745b9fa39774f1387273715a4167260")
    base_url = os.getenv("DASHSCOPE_BASE_URL",
                         "https://dashscope.aliyuncs.com/compatible-mode/v1")
    if not api_key:
        raise EnvironmentError("DASHSCOPE_API_KEY 未设置")
    return OpenAI(api_key=api_key, base_url=base_url)


def count_sentences(text: str) -> int:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return len([p for p in parts if p.strip()])


def _get_env_overlap(text: str) -> set:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) < 2:
        return set()
    first = {w for w in ENV_KEYWORDS if w in sentences[0].lower()}
    last  = {w for w in ENV_KEYWORDS if w in sentences[-1].lower()}
    return first & last


def _dynamic_max_tokens(original: str) -> int:
    """按原文长度动态计算 max_tokens（覆盖两个变体的输出）"""
    n = len(original.split())
    tokens = int(n * MAX_TOKENS_RATIO * 2)
    return max(MAX_TOKENS_FLOOR, min(tokens, MAX_TOKENS_CAP))


def parse_two_variants(response_text: str) -> Optional[tuple]:
    """
    解析模型返回的两个变体文本。
    支持 'Variant 1: ...' 格式，回退支持 '1. ...' 编号格式。
    """
    m1 = re.search(
        r'Variant\s*1\s*:\s*(.+?)(?=Variant\s*2\s*:|\Z)',
        response_text, re.DOTALL | re.IGNORECASE
    )
    m2 = re.search(
        r'Variant\s*2\s*:\s*(.+?)(?=Variant\s*3\s*:|\Z)',
        response_text, re.DOTALL | re.IGNORECASE
    )
    if m1 and m2:
        v1 = m1.group(1).strip()
        v2 = m2.group(1).strip()
        if v1 and v2:
            return v1, v2

    # 回退：按编号行拆分
    lines   = [l.strip() for l in response_text.strip().split('\n') if l.strip()]
    blocks, current = [], []
    for line in lines:
        if re.match(r'^(?:Variant\s*\d+\s*:|\d+\.)\s*', line, re.IGNORECASE):
            if current:
                blocks.append(' '.join(current))
            current = [re.sub(r'^(?:Variant\s*\d+\s*:|\d+\.\s*)', '',
                               line, flags=re.IGNORECASE).strip()]
        else:
            current.append(line)
    if current:
        blocks.append(' '.join(current))

    if len(blocks) >= 2:
        return blocks[0].strip(), blocks[1].strip()

    return None


def validate_variant(text: str, original: str,
                     orig_sentence_count: int,
                     orig_words: int,
                     orig_env_overlap: set,
                     required_env: list) -> Optional[str]:
    """校验单个变体，返回 fail_reason 或 None（通过）"""
    ratio = len(text.split()) / max(orig_words, 1)
    if not (MIN_LEN_RATIO <= ratio <= MAX_LEN_RATIO):
        return f"长度比例异常({ratio:.2f})"

    v_sent = count_sentences(text)
    if v_sent != orig_sentence_count:
        return f"句数不一致(原{orig_sentence_count}句，变体{v_sent}句)"

    missing = [t for t in FIXED_TERMS
               if t in original.lower() and t not in text.lower()]
    if missing:
        return f"固定术语丢失{missing}"

    missing_env = [e for e in required_env if e not in text.lower()]
    if missing_env:
        return f"环境词不一致，缺少{missing_env}"

    if not orig_env_overlap:
        variant_overlap = _get_env_overlap(text)
        if variant_overlap:
            return f"首尾句环境词重复{variant_overlap}"

    return None


# ── 核心 API 调用 ─────────────────────────────────────────────────────

def call_api(
    client: OpenAI,
    original: str,
    temperature: float,
) -> Optional[tuple]:
    """
    单次调用返回 (variant1, variant2)。
    重试 MAX_RETRIES 次，耗尽后返回最后一次解析结果（不回退原文）。
    """
    user_content = USER_PROMPT_TMPL.format(original=original)
    max_tokens   = _dynamic_max_tokens(original)

    orig_sentence_count = count_sentences(original)
    orig_words          = len(original.split())
    orig_env_overlap    = _get_env_overlap(original)
    required_env        = [req for orig_env, req in ENV_PAIRS
                           if orig_env in original.lower()]

    last_parsed = None   # 保存最后一次解析结果，耗尽重试后的兜底

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw_text = resp.choices[0].message.content.strip()
            parsed   = parse_two_variants(raw_text)

            if parsed is None:
                logger.warning(f"无法解析两个变体，重试{attempt}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
                continue

            v1, v2       = parsed
            last_parsed  = (v1, v2)

            # 两个变体分别独立校验
            fail1 = validate_variant(v1, original, orig_sentence_count,
                                     orig_words, orig_env_overlap, required_env)
            fail2 = validate_variant(v2, original, orig_sentence_count,
                                     orig_words, orig_env_overlap, required_env)

            if fail1 or fail2:
                reasons = []
                if fail1: reasons.append(f"变体1: {fail1}")
                if fail2: reasons.append(f"变体2: {fail2}")
                logger.warning(f"{'; '.join(reasons)}，重试{attempt}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
                continue

            return v1, v2   # 两个变体均通过校验

        except Exception as e:
            logger.warning(f"API调用失败(attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)

    # 耗尽重试：使用最后一次解析结果（不回退原文，保证有句式变化）
    if last_parsed:
        logger.warning(
            f"已达最大重试次数({MAX_RETRIES})，使用最后一次生成结果（未通过校验）"
        )
        return last_parsed

    # API 完全无响应（极少发生）才返回 None
    logger.error(f"已达最大重试次数({MAX_RETRIES})且无任何生成，返回 None")
    return None


# ── 单条数据增强 ──────────────────────────────────────────────────────

def augment_one(
    client: OpenAI,
    item: dict,
    num_variants: int,
    line_idx: int,
) -> list:
    """
    返回 [原始条目, 变体1, 变体2, ...]
    输出格式与输入完全一致，不新增任何字段。
    """
    conversations = item.get("conversations", [])
    gpt_turn = next((c for c in conversations if c.get("from") == "gpt"), None)
    if gpt_turn is None:
        logger.warning(f"第{line_idx}条无 gpt 回复，跳过增强")
        return [item]

    original_text = gpt_turn["value"]
    results       = [item]

    # num_variants=2：单次调用
    if num_variants == 2:
        pair = call_api(client, original_text, temperature=0.80)

        if pair is None:
            logger.warning(f"第{line_idx}条 API 无响应，变体回退原文")
            pair = (original_text, original_text)

        for variant_text in pair:
            new_item = copy.deepcopy(item)
            for conv in new_item["conversations"]:
                if conv.get("from") == "gpt":
                    conv["value"] = variant_text
                    break
            results.append(new_item)

    else:
        # num_variants != 2：分批生成，每批出 2 个，取前 num_variants 个
        temperatures = [0.75, 0.85, 0.90, 0.80]
        collected    = []
        batch_idx    = 0
        while len(collected) < num_variants:
            temp = temperatures[batch_idx % len(temperatures)]
            pair = call_api(client, original_text, temperature=temp)
            if pair is None:
                collected.extend([original_text, original_text])
            else:
                collected.extend(pair)
            batch_idx += 1

        for variant_text in collected[:num_variants]:
            new_item = copy.deepcopy(item)
            for conv in new_item["conversations"]:
                if conv.get("from") == "gpt":
                    conv["value"] = variant_text
                    break
            results.append(new_item)

    return results


# ── 断点续传 ──────────────────────────────────────────────────────────

def load_progress(progress_file: Path) -> set:
    if progress_file.exists():
        return set(json.loads(progress_file.read_text()))
    return set()


def save_progress(progress_file: Path, done: set):
    progress_file.write_text(json.dumps(sorted(done)))


# ── 主函数 ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Caption 数据增强（Token 优化版 V2）")
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
    logger.info(f"预计输出 {total * (args.num_variants + 1)} 条")
    logger.info(f"API 调用次数: ~{total} 次（合并生成，原版需 {total*2} 次）")

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
            rows = augment_one(client, item, args.num_variants, idx + 1)
            return idx, rows, True
        except Exception as e:
            logger.error(f"第{idx+1}条处理异常: {e}")
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
                # 处理异常：写入原始条目，记录到 failed
                if isinstance(result, dict):
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                fail_f.write(
                    json.dumps({"line": idx + 1, "item": result},
                               ensure_ascii=False) + "\n"
                )
                failed += 1

    out_f.close()
    fail_f.close()

    out_lines = sum(1 for _ in open(out_path, encoding="utf-8"))
    logger.info("=" * 50)
    logger.info(f"完成！输出文件：{out_path}")
    logger.info(f"输出总条数：{out_lines}  （预期 {total * (args.num_variants + 1)}）")
    logger.info(f"成功处理：{completed} 条 | 失败回退：{failed} 条")
    if failed:
        logger.info(f"失败记录见：{failed_file}")


if __name__ == "__main__":
    main()