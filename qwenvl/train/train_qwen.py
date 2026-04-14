import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwenvl.train.trainer
from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen3VLForConditionalGeneration,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from qwenvl.data.data_qwen import make_supervised_data_module

from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Trainer

local_rank = None
TRAINABLE_KEYWORDS = ["merger", "patch_merger", "deepstack_merger", "lora_"]


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def get_visual_module(model):
    """兼容 Qwen3-VL / DeepSpeed / PEFT 包装"""
    m = model
    if hasattr(m, 'base_model'):
        m = m.base_model
    if hasattr(m, 'model'):
        m = m.model
    if hasattr(m, 'module'):
        m = m.module
    if hasattr(m, 'visual'):
        return m.visual
    elif hasattr(m, 'model') and hasattr(m.model, 'visual'):
        return m.model.visual
    else:
        raise AttributeError(f"Cannot find visual module in {type(m).__name__}")


# ══════════════════════════════════════════════════════════════════════
# GradientDebugCallback：修复 norm.weight 误报
#
# 原问题：LayerNorm weight（γ）初始值为 1.0，前几步梯度极小，
#   绝对差值 < 1e-9 被误判为"未更新"，实际上是正常现象。
#
# 修复方案：改用相对阈值 max(1e-9, 1e-7 × |初始值均值|)
#   - 对普通参数（初始值接近 0）：阈值仍约 1e-9，行为不变
#   - 对 norm.weight（初始值 ≈ 1.0）：阈值放宽至 1e-7，消除误报
#   - 对 norm.bias（初始值 = 0）：阈值仍约 1e-9，行为不变
# ══════════════════════════════════════════════════════════════════════
class GradientDebugCallback(TrainerCallback):
    """3步后检查可训练参数是否有变化（使用相对阈值，消除 norm.weight 误报）"""

    # 相对阈值系数：阈值 = max(ABS_FLOOR, REL_SCALE × |初始值均值|)
    ABS_FLOOR  = 1e-9   # 绝对下限，防止初始值为 0 时除零
    REL_SCALE  = 1e-7   # 相对系数，对 norm.weight(≈1.0) 阈值 ≈ 1e-7

    def __init__(self, model):
        self.model = model
        self.checked = False
        self.initial_params = {}

    def on_train_begin(self, args, state, control, **kwargs):
        if args.local_rank not in (-1, 0):
            return
        m = self.model
        if hasattr(m, 'module'):
            m = m.module
        count = 0
        for n, p in m.named_parameters():
            if p.requires_grad:
                self.initial_params[n] = p.detach().cpu().clone()
                count += 1
        print(f"[GradCheck] 记录了 {count} 个可训练参数初始值")

    def _threshold(self, name: str) -> float:
        """
        根据参数初始值大小动态决定判断阈值。
        norm.weight 初始值为 1.0，使用相对阈值 1e-7；
        其他参数初始值接近 0，退化为绝对阈值 1e-9。
        """
        if name not in self.initial_params:
            return self.ABS_FLOOR
        init_magnitude = self.initial_params[name].abs().mean().item()
        return max(self.ABS_FLOOR, self.REL_SCALE * init_magnitude)

    def on_step_end(self, args, state, control, **kwargs):
        if self.checked or args.local_rank not in (-1, 0):
            return
        if state.global_step < 3:
            return
        self.checked = True

        m = self.model
        if hasattr(m, 'module'):
            m = m.module

        print("\n[GradCheck] === 参数变化量检查（3 步后）===")
        changed, unchanged = [], []

        for n, p in m.named_parameters():
            if p.requires_grad and n in self.initial_params:
                diff      = (p.detach().cpu() - self.initial_params[n]).abs().max().item()
                threshold = self._threshold(n)
                if diff > threshold:
                    changed.append(f"  ✅ {n}: max_diff={diff:.2e} (threshold={threshold:.2e})")
                else:
                    unchanged.append(f"  ❌ {n}: max_diff={diff:.2e} (threshold={threshold:.2e})")

        for line in changed[:5]:
            print(line)
        if len(changed) > 5:
            print(f"  ... 共 {len(changed)} 个参数已更新")
        for line in unchanged[:5]:
            print(line)
        if len(unchanged) > 5:
            print(f"  ... 共 {len(unchanged)} 个参数未变化")

        if changed:
            print(f"  [OK] {len(changed)} 个参数正在更新，训练正常 ✅")
        else:
            print("  [WARN] 所有参数完全未变化，梯度未传到可训练参数")
        print("[GradCheck] =====================================\n")


def set_model(model_args, model):
    visual = get_visual_module(model)

    print(f"[DEBUG] tune_mm_vision={model_args.tune_mm_vision}")
    print(f"[DEBUG] tune_mm_mlp={model_args.tune_mm_mlp}")
    print(f"[DEBUG] tune_mm_llm={model_args.tune_mm_llm}")
    print(f"[DEBUG] use_lora={getattr(model_args, 'use_lora', False)}")
    print(f"[DEBUG] hasattr(visual, 'merger')={hasattr(visual, 'merger')}")
    print(f"[DEBUG] visual type={type(visual)}")

    # ── 第1步：处理 LLM ──────────────────────────────────────────────
    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        if hasattr(model, 'lm_head'):
            for p in model.lm_head.parameters():
                p.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        if hasattr(model, 'lm_head'):
            for p in model.lm_head.parameters():
                p.requires_grad = False

    # ── 第2步：处理视觉编码器 ────────────────────────────────────────
    if model_args.tune_mm_vision:
        for n, p in visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in visual.named_parameters():
            p.requires_grad = False

    # ── 第3步：解冻 merger（最后执行，保证优先级最高）────────────────
    merger = None
    if hasattr(visual, 'merger'):
        merger = visual.merger
    elif hasattr(visual, 'patch_merger'):
        merger = visual.patch_merger

    if merger is not None:
        if model_args.tune_mm_mlp:
            for n, p in merger.named_parameters():
                p.requires_grad = True
        else:
            for n, p in merger.named_parameters():
                p.requires_grad = False

    if hasattr(visual, 'deepstack_merger_list') and model_args.tune_mm_mlp:
        for n, p in visual.deepstack_merger_list.named_parameters():
            p.requires_grad = True


def apply_lora(model_args, model):
    """应用 LoRA 到 LLM 部分"""
    from peft import LoraConfig, get_peft_model, TaskType

    target_modules = [
        m.strip() for m in model_args.lora_target_modules.split(",")
    ]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=model_args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    rank0_print("=== LoRA 可训练参数统计 ===")
    model.print_trainable_parameters()

    # get_peft_model 后重新确保 merger 可训练
    visual = get_visual_module(model)
    merger = None
    if hasattr(visual, 'merger'):
        merger = visual.merger
    elif hasattr(visual, 'patch_merger'):
        merger = visual.patch_merger

    if merger is not None and model_args.tune_mm_mlp:
        for p in merger.parameters():
            p.requires_grad = True
        rank0_print("[INFO] merger requires_grad restored after LoRA wrapping")

    if hasattr(visual, 'deepstack_merger_list') and model_args.tune_mm_mlp:
        for m in visual.deepstack_merger_list:
            for p in m.parameters():
                p.requires_grad = True
        rank0_print("[INFO] deepstack_merger_list requires_grad restored")

    return model


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # ── 加载 Qwen3-VL ────────────────────────────────────────────────
    try:
        from transformers import Qwen3VLForConditionalGeneration
    except ImportError:
        raise ImportError(
            "当前 transformers 不支持 Qwen3VLForConditionalGeneration，"
            "请升级：pip install 'transformers>=4.52.0'"
        )

    rank0_print("Loading Qwen3-VL model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    )
    data_args.image_processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    ).image_processor
    data_args.model_type = "qwen3vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # 第一步：设置 requires_grad
    set_model(model_args, model)

    # 第二步：应用 LoRA（在 set_model 之后，LoRA 只作用于 LLM）
    use_lora = getattr(model_args, 'use_lora', False)
    if use_lora and model_args.tune_mm_llm:
        model = apply_lora(model_args, model)
        rank0_print("[INFO] LoRA applied to LLM")
    elif model_args.tune_mm_llm:
        rank0_print("[INFO] Full LLM fine-tuning (no LoRA)")

    # 验证可训练参数
    if torch.distributed.get_rank() == 0:
        trainable = [
            (n, p.shape, p.numel())
            for n, p in model.named_parameters() if p.requires_grad
        ]
        total_trainable = sum(x[2] for x in trainable)

        lora_params   = [(n, s, c) for n, s, c in trainable if 'lora_' in n]
        merger_params = [(n, s, c) for n, s, c in trainable if 'merger' in n]

        rank0_print(f"\n[INFO] ══════ 可训练参数汇总 ══════")
        rank0_print(f"  LoRA 参数:    {sum(x[2] for x in lora_params)/1e6:.1f}M "
                    f"({len(lora_params)} tensors)")
        rank0_print(f"  Merger 参数:  {sum(x[2] for x in merger_params)/1e6:.1f}M "
                    f"({len(merger_params)} tensors)")
        rank0_print(f"  合计:         {total_trainable/1e6:.1f}M")
        rank0_print(f"[INFO] ══════════════════════════════\n")

    if torch.distributed.get_rank() == 0:
        visual = get_visual_module(model)
        visual.print_trainable_parameters()
        if not use_lora:
            model.model.print_trainable_parameters()

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    grad_debug_callback = GradientDebugCallback(model)

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[grad_debug_callback],
        **data_module,
    )

    # 验证 checkpoint 完整性
    def is_valid_checkpoint(ckpt_path):
        has_state     = (ckpt_path / "trainer_state.json").exists()
        has_ds_latest = (ckpt_path / "latest").exists()
        has_scheduler = (ckpt_path / "scheduler.pt").exists()
        return has_state and has_ds_latest and has_scheduler

    checkpoints       = sorted(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    valid_checkpoints = [p for p in checkpoints if is_valid_checkpoint(p)]

    if valid_checkpoints:
        latest = str(valid_checkpoints[-1])
        logging.info(f"Valid checkpoint found: {latest}, resuming...")
        trainer.train(resume_from_checkpoint=latest)
    else:
        if checkpoints:
            logging.warning(
                f"Found {len(checkpoints)} incomplete checkpoint(s), starting from scratch"
            )
        trainer.train()

    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    source_path   = os.path.join(model_args.model_name_or_path, "chat_template.json")
    template_path = os.path.join(training_args.output_dir, "chat_template.json")
    shutil.copy2(source_path, template_path)

    # ── 保存 LoRA adapter 并合并为完整模型 ──────────────────────────
    if use_lora and training_args.local_rank in (-1, 0):
        lora_save_path = os.path.join(training_args.output_dir, "lora_adapter")
        model.save_pretrained(lora_save_path)
        rank0_print(f"[INFO] LoRA adapter saved to {lora_save_path}")

        rank0_print("[INFO] Merging LoRA weights into base model (on GPU)...")
        merged_model = model.merge_and_unload()

        merged_save_path = os.path.join(training_args.output_dir, "merged_model")
        merged_model.save_pretrained(merged_save_path)
        tokenizer.save_pretrained(merged_save_path)
        data_args.image_processor.save_pretrained(merged_save_path)
        rank0_print(f"[INFO] Merged model saved to {merged_save_path}")
        rank0_print(f"[INFO] 推理时使用: --model_path {merged_save_path}")

    rank0_print("[INFO] Training complete.")


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")