"""
qwenvl/train/train_qwen.py  (Stage 1 修复版)

核心改动:
  1. 加载 RSCaptionModel 代替 Qwen3VLForConditionalGeneration
  2. register_semantic_slot / register_style_prefix 替换为
     build_semantic_slot / build_style_prefix + model.attach_modules()
  3. GradientDebugCallback 增加 style_prefix / semantic_slot 专项检查
"""

import logging
import os
import pathlib
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch
import transformers

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwenvl.train.trainer
from qwenvl.train.trainer import replace_qwen2_vl_attention_class

from transformers import TrainerCallback
from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.train.argument import ModelArguments, DataArguments, TrainingArguments

# ★ Stage 1: 使用 RSCaptionModel
from qwenvl.model.rs_caption_model import RSCaptionModel

from transformers import AutoTokenizer, AutoProcessor, Trainer

logger = logging.getLogger(__name__)
local_rank = None

TRAINABLE_KEYWORDS = [
    "merger", "patch_merger", "deepstack_merger",
    "lora_", "semantic_slot", "style_prefix",
]


def rank0_print(*args):
    if local_rank in (-1, 0):
        print(*args)


# ══════════════════════════════════════════════════════════════════════
# 模型结构工具
# ══════════════════════════════════════════════════════════════════════

def get_visual_module(model):
    """
    穿透 DDP / PeftModel 包装,返回 Qwen3VLVisionModel。

    可能的层级 (从外到内):
      DDP:        .module
      PeftModel:  .base_model (= LoraModel) → .model (= RSCaptionModel)
      PreTrainedModel.base_model (property): 返回 self.model (= Qwen3VLModel)
      RSCaptionModel: .model (= Qwen3VLModel) → .visual
    """
    m = model

    # 1) 剥 DDP
    if hasattr(m, "module"):
        m = m.module

    # 2) 剥 PEFT 或 PreTrainedModel.base_model 属性
    if hasattr(m, "base_model"):
        m = m.base_model
        # LoraModel.model = RSCaptionModel, 需要再下一层才到 visual.
        # 判定:LoraModel 没有直接 .visual, Qwen3VLModel 有直接 .visual。
        # 若当前层无 .visual 但有 .model, 说明还在包装层,继续剥。
        if hasattr(m, "model") and not hasattr(m, "visual"):
            m = m.model

    # 3) 此时 m 可能是:
    #    - Qwen3VLModel (有 .visual 直接子模块)
    #    - RSCaptionModel / Qwen3VLForConditionalGeneration (有 .model.visual)
    if hasattr(m, "visual"):
        return m.visual
    if hasattr(m, "model") and hasattr(m.model, "visual"):
        return m.model.visual

    raise AttributeError(
        f"Cannot find visual in {type(m).__name__}. "
        f"Children: {[n for n, _ in m.named_children()]}"
    )


def get_rs_caption_model(model) -> RSCaptionModel:
    """
    穿透 PEFT / DDP 包装,返回 RSCaptionModel 实例。
    在 attach_modules 等需要访问 RSCaptionModel 原生属性时使用。
    """
    m = model
    if hasattr(m, "module"):     m = m.module
    if hasattr(m, "base_model"): m = m.base_model
    if hasattr(m, "model") and isinstance(m.model, RSCaptionModel):
        return m.model
    if isinstance(m, RSCaptionModel):
        return m
    # 兜底: 递归找
    for attr in ["model"]:
        child = getattr(m, attr, None)
        if isinstance(child, RSCaptionModel):
            return child
    raise AttributeError(f"Cannot find RSCaptionModel in {type(model).__name__}")


# ══════════════════════════════════════════════════════════════════════
# Output dir 管理
# ══════════════════════════════════════════════════════════════════════

def is_valid_checkpoint(p):
    return (p / "trainer_state.json").exists() and (
        (p / "latest").exists() or (p / "scheduler.pt").exists()
    )


def prepare_output_dir(training_args):
    output_dir = training_args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    all_ckpts = sorted(pathlib.Path(output_dir).glob("checkpoint-*"))

    if training_args.resume_training:
        valid = sorted(
            [p for p in all_ckpts if is_valid_checkpoint(p)],
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if valid:
            latest = str(valid[-1])
            rank0_print(f"\n{'='*60}\n[续训] {latest}\n{'='*60}\n")
            return latest
        rank0_print("\n[续训] 无有效 checkpoint,从头开始\n")
        return None
    else:
        if all_ckpts:
            ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = f"{output_dir.rstrip('/')}_backup_{ts}"
            shutil.move(output_dir, backup)
            os.makedirs(output_dir, exist_ok=True)
            rank0_print(f"\n{'='*60}\n[全新训练] 旧目录已备份: {backup}\n{'='*60}\n")
        return None


# ══════════════════════════════════════════════════════════════════════
# GradientDebugCallback (增强版,含 Style/Slot 专项检查)
# ══════════════════════════════════════════════════════════════════════

class GradientDebugCallback(TrainerCallback):
    ABS_FLOOR = 1e-9
    REL_SCALE = 1e-7

    def __init__(self, model):
        self.model          = model
        self.checked        = False
        self.initial_params = {}

    def on_train_begin(self, args, state, control, **kwargs):
        if args.local_rank not in (-1, 0):
            return
        m = self.model
        if hasattr(m, "module"): m = m.module
        count = 0
        for n, p in m.named_parameters():
            if p.requires_grad:
                self.initial_params[n] = p.detach().cpu().clone()
                count += 1
        print(f"[GradCheck] 记录了 {count} 个可训练参数")

    def _thr(self, n):
        if n not in self.initial_params:
            return self.ABS_FLOOR
        return max(
            self.ABS_FLOOR,
            self.REL_SCALE * self.initial_params[n].abs().mean().item(),
        )

    def on_step_end(self, args, state, control, **kwargs):
        if self.checked or args.local_rank not in (-1, 0) or state.global_step < 3:
            return
        self.checked = True
        m = self.model
        if hasattr(m, "module"): m = m.module

        changed, unchanged = [], []
        for n, p in m.named_parameters():
            if p.requires_grad and n in self.initial_params:
                diff = (p.detach().cpu() - self.initial_params[n]).abs().max().item()
                (changed if diff > self._thr(n) else unchanged).append(
                    (n, diff)
                )

        style_chg = [(n, d) for n, d in changed   if "style_prefix"  in n]
        slot_chg  = [(n, d) for n, d in changed   if "semantic_slot" in n]
        style_unc = [(n, d) for n, d in unchanged if "style_prefix"  in n]
        slot_unc  = [(n, d) for n, d in unchanged if "semantic_slot" in n]

        print(f"\n[GradCheck] Step {state.global_step}")
        print(f"  总计: ✅ {len(changed)} 更新  ❌ {len(unchanged)} 未变化")
        print(f"  StylePrefix  更新: {len(style_chg):2d}  未变化: {len(style_unc):2d}")
        print(f"  SemanticSlot 更新: {len(slot_chg):2d}  未变化: {len(slot_unc):2d}")

        # ★ 核心告警
        if style_unc and any("style_embeds" in n for n, _ in style_unc):
            print("  ⚠️  [WARN] style_embeds 未更新 → Bug #4 修复失败!")
        if slot_unc and any("slot_init" in n for n, _ in slot_unc):
            print("  ⚠️  [WARN] slot_init 未更新!")

        for n, d in (changed[:3] + unchanged[:3]):
            mark = "✅" if (n, d) in changed else "❌"
            print(f"  {mark} {n}: {d:.2e}")
        print("[GradCheck] ================\n")


# ══════════════════════════════════════════════════════════════════════
# 模型参数配置
# ══════════════════════════════════════════════════════════════════════

def set_model(model_args, model):
    visual = get_visual_module(model)
    rank0_print(
        f"[DEBUG] tune_mm_vision={model_args.tune_mm_vision} "
        f"tune_mm_mlp={model_args.tune_mm_mlp} tune_mm_llm={model_args.tune_mm_llm}"
    )

    for p in model.model.parameters():
        p.requires_grad = model_args.tune_mm_llm
    if hasattr(model, "lm_head"):
        for p in model.lm_head.parameters():
            p.requires_grad = model_args.tune_mm_llm
    for p in visual.parameters():
        p.requires_grad = model_args.tune_mm_vision

    merger = getattr(visual, "merger", None) or getattr(visual, "patch_merger", None)
    if merger:
        for p in merger.parameters():
            p.requires_grad = model_args.tune_mm_mlp
    if hasattr(visual, "deepstack_merger_list") and model_args.tune_mm_mlp:
        for p in visual.deepstack_merger_list.parameters():
            p.requires_grad = True


def apply_lora(model_args, model):
    from peft import LoraConfig, get_peft_model, TaskType
    cfg = LoraConfig(
        task_type       = TaskType.CAUSAL_LM,
        r               = model_args.lora_r,
        lora_alpha      = model_args.lora_alpha,
        target_modules  = [m.strip() for m in model_args.lora_target_modules.split(",")],
        lora_dropout    = model_args.lora_dropout,
        bias            = "none",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    visual = get_visual_module(model)
    merger = getattr(visual, "merger", None) or getattr(visual, "patch_merger", None)
    if merger and model_args.tune_mm_mlp:
        for p in merger.parameters():
            p.requires_grad = True
    if hasattr(visual, "deepstack_merger_list") and model_args.tune_mm_mlp:
        for m in visual.deepstack_merger_list:
            for p in m.parameters():
                p.requires_grad = True
    return model


# ══════════════════════════════════════════════════════════════════════
# ★ Stage 1: 模块构建 (不含任何 monkey-patch)
# ══════════════════════════════════════════════════════════════════════

def build_semantic_slot(model_args, dim: int):
    """创建 SemanticSlotModule 实例 (不挂载)"""
    if not getattr(model_args, "use_semantic_slot", False):
        return None

    from qwenvl.model.semantic_slot import SemanticSlotModule
    slot = SemanticSlotModule(
        num_slots         = model_args.slot_num_slots,
        dim               = dim,
        num_iterations    = model_args.slot_num_iterations,
        use_position_info = model_args.use_position_info,
        use_count_head    = getattr(model_args, "use_count_head", True),
    )
    for p in slot.parameters():
        p.requires_grad = True
    return slot


def build_style_prefix(model_args, dim: int, tokenizer=None, embed_fn=None):
    """创建 StylePrefixModule 实例 (不挂载)"""
    if not getattr(model_args, "use_style_prefix", False):
        return None

    from qwenvl.model.style_prefix import StylePrefixModule
    style = StylePrefixModule(
        num_style_tokens    = model_args.style_num_tokens,
        dim                 = dim,
        use_slot_attention  = getattr(model_args, "style_use_slot_attention", True),
    )

    init_str = getattr(model_args, "style_init_sentences", "")
    if init_str.strip() and tokenizer is not None and embed_fn is not None:
        sentences = [s.strip() for s in init_str.split(";") if s.strip()]
        try:
            style.initialize_from_sentences(sentences, tokenizer, embed_fn)
        except Exception as e:
            rank0_print(f"[StylePrefix] 句子初始化失败,使用随机初始化: {e}")

    for p in style.parameters():
        p.requires_grad = True
    return style


# ══════════════════════════════════════════════════════════════════════
# 主训练函数
# ══════════════════════════════════════════════════════════════════════

def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    resume_from = None
    if training_args.local_rank in (-1, 0):
        resume_from = prepare_output_dir(training_args)
    if torch.distributed.is_initialized():
        obj = [resume_from or ""]
        torch.distributed.broadcast_object_list(obj, src=0)
        resume_from = obj[0] if obj[0] else None

    # ── ★ Stage 1: 加载 RSCaptionModel ──
    rank0_print("Loading RSCaptionModel (inherits Qwen3-VL)...")
    model = RSCaptionModel.from_pretrained(
        model_args.model_name_or_path,
        cache_dir            = training_args.cache_dir,
        attn_implementation  = attn_implementation,
        torch_dtype          = torch.bfloat16 if training_args.bf16 else None,
    )

    data_args.image_processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path
    ).image_processor
    data_args.model_type = "qwen3vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        model.get_input_embeddings().register_forward_hook(
            lambda m, i, o: o.requires_grad_(True)
        )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir        = training_args.cache_dir,
        model_max_length = training_args.model_max_length,
        padding_side     = "right",
        use_fast         = False,
    )

    set_model(model_args, model)

    use_lora = getattr(model_args, "use_lora", False)
    if use_lora and model_args.tune_mm_llm:
        model = apply_lora(model_args, model)

    # ── 获取模型维度 ──
    rs_model = get_rs_caption_model(model)
    dim = rs_model.config.text_config.hidden_size

    # ── ★ Stage 1: 构建模块 ──
    slot_module  = build_semantic_slot(model_args, dim)
    style_module = build_style_prefix(
        model_args, dim,
        tokenizer = tokenizer,
        embed_fn  = rs_model.model.language_model.embed_tokens,
    )

    # ── ★ Stage 1: 通过 attach_modules 挂载 (slot 先于 style) ──
    rs_model.attach_modules(slot_module=slot_module, style_module=style_module)

    # ── 打印参数汇总 ──
    _is_rank0 = (
        not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    )
    if _is_rank0:
        trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
        total     = sum(c for _, c in trainable)
        rank0_print(f"\n[INFO] ══════ 可训练参数汇总 ══════")
        rank0_print(f"  LoRA         : {sum(c for n,c in trainable if 'lora_'         in n)/1e6:.1f}M")
        rank0_print(f"  Merger       : {sum(c for n,c in trainable if 'merger'        in n)/1e6:.1f}M")
        rank0_print(f"  SemanticSlot : {sum(c for n,c in trainable if 'semantic_slot' in n)/1e6:.2f}M")
        rank0_print(f"  StylePrefix  : {sum(c for n,c in trainable if 'style_prefix'  in n)/1e6:.4f}M")
        rank0_print(f"  ViT          : {sum(c for n,c in trainable if 'visual.blocks' in n)/1e6:.1f}M")
        rank0_print(f"  合计         : {total/1e6:.1f}M")
        rank0_print(f"[INFO] ══════════════════════════════\n")
        if slot_module:  slot_module.print_params()
        if style_module: style_module.print_params()
        get_visual_module(model).print_trainable_parameters()

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = Trainer(
        model            = model,
        processing_class = tokenizer,
        args             = training_args,
        callbacks        = [GradientDebugCallback(model)],
        **data_module,
    )

    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    src = os.path.join(model_args.model_name_or_path, "chat_template.json")
    dst = os.path.join(training_args.output_dir, "chat_template.json")
    if os.path.exists(src):
        shutil.copy2(src, dst)

    if use_lora and training_args.local_rank in (-1, 0):
        lora_path = os.path.join(training_args.output_dir, "lora_adapter")
        model.save_pretrained(lora_path)
        rank0_print(f"[INFO] LoRA → {lora_path}")
        merged = model.merge_and_unload()
        mp = os.path.join(training_args.output_dir, "merged_model")
        merged.save_pretrained(mp)
        tokenizer.save_pretrained(mp)
        data_args.image_processor.save_pretrained(mp)
        rank0_print(f"[INFO] Merged → {mp}")

    rank0_print("[INFO] Training complete.")


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")