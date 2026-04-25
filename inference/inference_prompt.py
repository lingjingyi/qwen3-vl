"""
inference/generate_caption.py  (B2 兼容版 + BLEU 修复版)

本次修复:
  ★ 修复 #1: 使用 RSCaptionModel (B2 方式下 slot/style 通过 forward 直接调用)
  ★ 修复 #2: 去掉训练时从未见过的 FORMAT_SUFFIX, 避免分布漂移
  ★ 修复 #3: postprocess 默认 max_sentences=50 (即不截), 原 5 句截断导致 brevity penalty
  ★ 修复 #4: max_new_tokens=200→256, min_new_tokens=20→80 (GT 平均 132 词)
  ★ 修复 #5: 移除 score() 候选重排 (其奖励短文本, 与 GT 趋势相反), 直接用 beam-1 输出
  ★ 修复 #6: 正确用 attach_modules 挂载 slot/style
"""

import os, re, json, argparse, logging, sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

# 确保能 import 到 qwenvl
_proj_root = Path(__file__).parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from qwenvl.model.rs_caption_model import RSCaptionModel
from qwenvl.model.semantic_slot    import SemanticSlotModule
from qwenvl.model.style_prefix     import StylePrefixModule


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ★ Fix #2: 空字符串, 训练时没见过的指令不再追加
FORMAT_SUFFIX = ""


# ══════════════════════════════════════════════════════════════════════
# Prompt / 后处理
# ══════════════════════════════════════════════════════════════════════

def extract_prompt(sample: dict, add_suffix: bool = True) -> str:
    for turn in sample.get("conversations", []):
        if turn.get("from") == "human":
            v = re.sub(r"<image>\n?", "", turn.get("value", "")).strip()
            if v:
                return v + (FORMAT_SUFFIX if add_suffix else "")
    return (
        "Please provide a detailed description of this remote sensing image."
        + (FORMAT_SUFFIX if add_suffix else "")
    )


def postprocess(text: str, max_sentences: int = 50) -> str:
    """
    ★ Fix #3: max_sentences 默认 5→50, 实际等于不截.
    GT 平均 8-12 句, 原 5 句截断严重损失 BLEU.
    """
    # 清 markdown
    text = re.sub(r'\*{1,3}([^*\n]+)\*{1,3}', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    lines, parts, buf = text.split('\n'), [], ""
    for line in lines:
        line = line.strip()
        if not line:
            if buf:
                parts.append(buf.strip()); buf = ""
            continue
        m = re.match(r'^[-*•·]\s+(.+)', line) or re.match(r'^\d+[.)]\s+(.+)', line)
        if m:
            c = m.group(1).strip()
            if c and c[-1] not in '.!?':
                c += '.'
            if buf:
                parts.append(buf.strip()); buf = ""
            parts.append(c)
        else:
            buf = (buf + " " + line).strip() if buf else line
    if buf:
        parts.append(buf.strip())

    text = ' '.join(parts)
    text = re.sub(r' {2,}', ' ', text).strip()
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    return ' '.join(sents[:max_sentences])


# ══════════════════════════════════════════════════════════════════════
# 从 checkpoint 加载 slot / style 模块
# ══════════════════════════════════════════════════════════════════════

def _find_merger_weights(model_path: str) -> Optional[Path]:
    """
    依次查找 merger_weights.pt:
      1. model_path 本身
      2. model_path 的 parent 目录
      3. parent 下最新的 checkpoint-* 目录
    返回第一个存在的路径.
    """
    mp = Path(model_path)

    candidates = [
        mp / "merger_weights.pt",
        mp.parent / "merger_weights.pt",
    ]

    # 最新 checkpoint 目录
    parent_cks = []
    if mp.parent.exists():
        parent_cks = sorted(
            [p for p in mp.parent.glob("checkpoint-*") if p.is_dir()],
            key=lambda p: (int(p.name.split("-")[-1])
                           if p.name.split("-")[-1].isdigit() else 0),
            reverse=True,
        )
    for ck in parent_cks[:3]:
        candidates.append(ck / "merger_weights.pt")

    for c in candidates:
        if c.exists():
            logger.info(f"[weights] 找到 merger_weights.pt: {c}")
            return c

    logger.warning(
        f"[weights] 未找到 merger_weights.pt, 已搜索: "
        f"{[str(c) for c in candidates[:5]]}"
    )
    return None


def _strip_prefix(key: str, module_name: str) -> str:
    """剥离 state_dict key 的包装前缀"""
    for pfx in [
        f"base_model.model.model.{module_name}.",
        f"base_model.model.{module_name}.",
        f"model.model.{module_name}.",
        f"model.{module_name}.",
        f"{module_name}.",
    ]:
        if key.startswith(pfx):
            return key[len(pfx):]
    return key


def build_slot_from_ckpt(
    model_path: str, dtype: torch.dtype
) -> Optional[SemanticSlotModule]:
    """从 checkpoint 的 merger_weights.pt 构造 SemanticSlotModule"""
    wp = _find_merger_weights(model_path)
    if wp is None:
        return None

    sd = torch.load(wp, map_location="cpu", weights_only=True)
    ssd = {k: v for k, v in sd.items() if "semantic_slot" in k}
    if not ssd:
        logger.info("[SemanticSlot] checkpoint 里无 semantic_slot 权重")
        return None

    # 从 slot_init 推断 num_slots 和 dim
    ik = next((k for k in ssd if "slot_init" in k), None)
    if ik is None:
        logger.warning("[SemanticSlot] 无 slot_init, 跳过")
        return None
    ns, dim = ssd[ik].shape

    # 从权重自动判断是否有 count_head
    has_count = any("count_head" in k for k in ssd)

    sm = SemanticSlotModule(
        num_slots         = ns,
        dim               = dim,
        num_iterations    = 2,
        use_position_info = True,
        use_count_head    = has_count,
    )

    clean_sd = {_strip_prefix(k, "semantic_slot"): v for k, v in ssd.items()}
    missing, unexpected = sm.load_state_dict(clean_sd, strict=False)

    if missing:
        logger.warning(f"[SemanticSlot] missing ({len(missing)}): {missing[:5]}")
    if unexpected:
        logger.warning(f"[SemanticSlot] unexpected ({len(unexpected)}): {unexpected[:5]}")

    sm = sm.to(dtype=dtype, device="cuda").eval()
    for p in sm.parameters():
        p.requires_grad = False
    logger.info(
        f"[SemanticSlot] ✅ num_slots={ns} dim={dim} count_head={has_count}"
    )
    return sm


def build_style_from_ckpt(
    model_path: str, dtype: torch.dtype,
    slot_module: Optional[SemanticSlotModule] = None,
) -> Optional[StylePrefixModule]:
    """从 checkpoint 的 merger_weights.pt 构造 StylePrefixModule"""
    wp = _find_merger_weights(model_path)
    if wp is None:
        return None

    sd = torch.load(wp, map_location="cpu", weights_only=True)
    ssd = {k: v for k, v in sd.items() if "style_prefix" in k}
    if not ssd:
        logger.info("[StylePrefix] checkpoint 里无 style_prefix 权重")
        return None

    ek = next((k for k in ssd if "style_embeds" in k), None)
    if ek is None:
        return None
    K, dim = ssd[ek].shape

    has_slot_attn = any("slot_cross" in k or "slot_gate" in k for k in ssd)

    sp = StylePrefixModule(
        num_style_tokens   = K,
        dim                = dim,
        use_slot_attention = has_slot_attn,
    )

    clean_sd = {_strip_prefix(k, "style_prefix"): v for k, v in ssd.items()}
    missing, unexpected = sp.load_state_dict(clean_sd, strict=False)

    if missing:
        logger.warning(f"[StylePrefix] missing ({len(missing)}): {missing[:5]}")
    if unexpected:
        logger.warning(f"[StylePrefix] unexpected ({len(unexpected)}): {unexpected[:5]}")

    sp = sp.to(dtype=dtype, device="cuda").eval()
    for p in sp.parameters():
        p.requires_grad = False
    logger.info(
        f"[StylePrefix] ✅ K={K} dim={dim} slot_attn={has_slot_attn}"
    )
    return sp


# ══════════════════════════════════════════════════════════════════════
# CaptionGenerator
# ══════════════════════════════════════════════════════════════════════

class CaptionGenerator:
    def __init__(
        self, model_path, torch_dtype=torch.bfloat16,
        load_slot=True, load_style=True,
    ):
        logger.info(f"Loading RSCaptionModel from {model_path}")

        # ★ Fix #1: 用 RSCaptionModel (B2 forward 里会主动调 slot 的 refine)
        self.model = RSCaptionModel.from_pretrained(
            model_path,
            torch_dtype         = torch_dtype,
            device_map          = "auto",
            attn_implementation = "flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

        # 先 slot 后 style (slot 的 _last_slots 是 style 的输入)
        slot_module  = build_slot_from_ckpt(model_path, torch_dtype) if load_slot else None
        style_module = (build_style_from_ckpt(model_path, torch_dtype, slot_module)
                        if load_style else None)

        # ★ Fix #6: 用 attach_modules (B2 方式), 不再调已废弃的 register_hooks
        if slot_module is not None or style_module is not None:
            self.model.attach_modules(
                slot_module  = slot_module,
                style_module = style_module,
            )
            logger.info("[Model] attach_modules 完成")
        else:
            logger.info("[Model] 未加载 slot/style, 运行纯 base")

        self.model.eval()
        self.model.config.use_cache = True   # 推理时开 cache

        logger.info("Model ready.")

    # ─────────────────────────────────
    # 输入构造
    # ─────────────────────────────────
    def _inp_single(self, images, prompt):
        content = [{"type":"image","image":img} for img in images] + \
                  [{"type":"text","text":prompt}]
        text = self.processor.apply_chat_template(
            [{"role":"user","content":content}],
            tokenize=False, add_generation_prompt=True,
        )
        return self.processor(
            text=text, images=images, return_tensors="pt"
        ).to(self.model.device)

    def _inp_batch(self, bimgs, bprompts):
        texts, aimgs = [], []
        for imgs, p in zip(bimgs, bprompts):
            content = [{"type":"image","image":img} for img in imgs] + \
                      [{"type":"text","text":p}]
            texts.append(self.processor.apply_chat_template(
                [{"role":"user","content":content}],
                tokenize=False, add_generation_prompt=True,
            ))
            aimgs.extend(imgs)
        return self.processor(
            text=texts, images=aimgs,
            return_tensors="pt", padding=True,
        ).to(self.model.device)

    @staticmethod
    def _extract_response(t):
        """从 chat-formatted decoded text 里提取 assistant 回复部分"""
        pfx = "assistant\n"
        return t.split(pfx)[-1].strip() if pfx in t else t.strip()

    # ─────────────────────────────────
    # 生成
    # ─────────────────────────────────
    def _gen(self, inputs, max_new, min_new, num_beams, length_penalty,
             no_repeat_ngram, num_return):
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens       = max_new,
                min_new_tokens       = min_new,
                num_beams            = num_beams,
                length_penalty       = length_penalty,
                no_repeat_ngram_size = no_repeat_ngram,
                num_return_sequences = num_return,
                do_sample            = False,
                early_stopping       = True,
            )
        return [
            self._extract_response(t)
            for t in self.processor.batch_decode(
                out, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        ]

    def batch_generate(
        self, samples, id_key="id",
        max_new_tokens=256, min_new_tokens=80,
        num_beams=5, length_penalty=1.0,
        no_repeat_ngram_size=3, num_return_sequences=1,
        batch_size=1, output_file=None, add_suffix=False,
    ):
        # 准备输入
        valid = []
        for s in samples:
            paths = s.get("images") or s.get("image")
            if not paths:
                continue
            if not isinstance(paths, list):
                paths = [paths]
            valid.append({
                **s,
                "_paths":  paths,
                "_prompt": extract_prompt(s, add_suffix),
                id_key:    s.get(id_key, str(len(valid))),
            })

        logger.info(f"有效样本: {len(valid)}")
        if valid:
            logger.info(f"示例 prompt: {valid[0]['_prompt']}")

        results = []

        for start in tqdm(range(0, len(valid), batch_size), desc="Generating"):
            batch = valid[start:start + batch_size]
            try:
                bimgs    = [[Image.open(p).convert("RGB") for p in s["_paths"]]
                            for s in batch]
                bprompts = [s["_prompt"] for s in batch]

                if len(batch) == 1:
                    raw = self._gen(
                        self._inp_single(bimgs[0], bprompts[0]),
                        max_new_tokens, min_new_tokens, num_beams,
                        length_penalty, no_repeat_ngram_size,
                        num_return_sequences,
                    )
                else:
                    raw = self._gen(
                        self._inp_batch(bimgs, bprompts),
                        max_new_tokens, min_new_tokens, num_beams,
                        length_penalty, no_repeat_ngram_size, 1,
                    )

                # ★ Fix #5: 直接取第一条 (num_return_sequences=1 时就是 beam-1)
                # 移除原 score() 重排 (它惩罚长文本, 导致 BLEU 低)
                caps = [postprocess(c) for c in raw]

                # 若 num_return_sequences>1 且 batch_size=1, 仍按原逻辑取第一个
                if len(batch) == 1 and len(caps) > 1:
                    caps = [caps[0]]

                for s, cap in zip(batch, caps):
                    clean = {k: v for k, v in s.items()
                             if k not in ("_paths", "_prompt")}
                    result = {id_key: s[id_key], "prediction": cap, **clean}
                    results.append(result)
                    if output_file:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")

            except Exception as e:
                import traceback
                logger.error(f"batch [{start}] 失败: {e}\n{traceback.format_exc()}")
                # 单条重试
                for s in batch:
                    try:
                        imgs = [Image.open(p).convert("RGB") for p in s["_paths"]]
                        raw  = self._gen(
                            self._inp_single(imgs, s["_prompt"]),
                            max_new_tokens, min_new_tokens, num_beams,
                            length_penalty, no_repeat_ngram_size, 1,
                        )
                        cap = postprocess(raw[0])
                        clean = {k: v for k, v in s.items()
                                 if k not in ("_paths", "_prompt")}
                        r = {id_key: s[id_key], "prediction": cap, **clean}
                        results.append(r)
                        if output_file:
                            with open(output_file, "a", encoding="utf-8") as f:
                                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    except Exception as e2:
                        logger.error(f"重试 {s[id_key]} 失败: {e2}")

        return results


def load_samples(path):
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",           required=True,
                        help="merged_model 目录 (含 config.json + model.safetensors). "
                             "slot/style 权重从 parent 的 checkpoint-XXX 自动找.")
    parser.add_argument("--input_file",           required=True)
    parser.add_argument("--output_file",          default="predictions.jsonl")
    parser.add_argument("--id_key",               default="id")
    # ★ Fix #4: 更大的 token 预算
    parser.add_argument("--max_new_tokens",       type=int,   default=256)
    parser.add_argument("--min_new_tokens",       type=int,   default=80)
    # ★ Fix: 与 RSGPT 对齐
    parser.add_argument("--num_beams",            type=int,   default=5)
    parser.add_argument("--length_penalty",       type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int,   default=3)
    parser.add_argument("--num_return_sequences", type=int,   default=1)
    parser.add_argument("--batch_size",           type=int,   default=1)
    parser.add_argument("--no_slot",              action="store_true")
    parser.add_argument("--no_style",             action="store_true")
    # ★ Fix #2: 默认不加 suffix, 要加才显式 --format_suffix
    parser.add_argument("--format_suffix",        action="store_true",
                        help="是否追加 FORMAT_SUFFIX 到 prompt (默认不加, 与训练对齐)")
    args = parser.parse_args()

    if args.num_return_sequences > args.num_beams:
        args.num_return_sequences = args.num_beams

    logger.info("=" * 60)
    logger.info(f"  model_path  : {args.model_path}")
    logger.info(f"  input_file  : {args.input_file}")
    logger.info(f"  output_file : {args.output_file}")
    logger.info(f"  load_slot   : {not args.no_slot}")
    logger.info(f"  load_style  : {not args.no_style}")
    logger.info(f"  format_suffix: {args.format_suffix}")
    logger.info(f"  max_new_tokens: {args.max_new_tokens}")
    logger.info(f"  min_new_tokens: {args.min_new_tokens}")
    logger.info(f"  num_beams     : {args.num_beams}")
    logger.info("=" * 60)

    # 若 output_file 已存在, 清空 (避免追加到上次的结果)
    if os.path.exists(args.output_file):
        logger.warning(f"[Output] {args.output_file} 已存在, 清空")
        os.remove(args.output_file)

    samples = load_samples(args.input_file)
    logger.info(f"Loaded {len(samples)} samples")

    gen = CaptionGenerator(
        args.model_path,
        load_slot  = not args.no_slot,
        load_style = not args.no_style,
    )
    gen.batch_generate(
        samples              = samples,
        id_key               = args.id_key,
        max_new_tokens       = args.max_new_tokens,
        min_new_tokens       = args.min_new_tokens,
        num_beams            = args.num_beams,
        length_penalty       = args.length_penalty,
        no_repeat_ngram_size = args.no_repeat_ngram_size,
        num_return_sequences = args.num_return_sequences,
        batch_size           = args.batch_size,
        output_file          = args.output_file,
        add_suffix           = args.format_suffix,
    )
    logger.info(f"Done → {args.output_file}")


if __name__ == "__main__":
    main()