import os
from typing import Optional
import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from transformers import Trainer
from transformers.cache_utils import Cache

try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel, Qwen3VLModel
except ImportError:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VisionTransformerPretrainedModel as Qwen3VLVisionModel, Qwen3VLModel,
    )
try:
    from transformers.trainer import ALL_LAYERNORM_LAYERS, get_parameter_names
except ImportError:
    from transformers.trainer import get_parameter_names
    ALL_LAYERNORM_LAYERS = (nn.LayerNorm, nn.RMSNorm)


def _flash_attention_forward(q, k, v, attn_mask, query_length, is_causal, dropout=0.0,
                              position_ids=None, softmax_scale=None, sliding_window=None,
                              use_top_left_mask=False, softcap=None, deterministic=None,
                              cu_seq_lens_q=None, cu_seq_lens_k=None, max_length_q=None,
                              max_length_k=None, target_dtype=None, **kwargs):
    assert q.size(0) == k.size(0) == v.size(0) == 1
    q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)
    cu = attn_mask
    with torch.no_grad():
        ms = max(cu[i+1]-cu[i] for i in range(cu.size(0)-1)).item()
    causal = is_causal if not use_top_left_mask else (is_causal and query_length != 1)
    kw = {"softcap": softcap} if softcap else {}
    out = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                  max_seqlen_q=ms, max_seqlen_k=ms,
                                  dropout_p=dropout, softmax_scale=softmax_scale, causal=causal, **kw)
    return out.unsqueeze(0)


def _update_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions):
    return attention_mask


def replace_qwen2_vl_attention_class():
    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl
        modeling_qwen3_vl._flash_attention_forward = _flash_attention_forward
        modeling_qwen3_vl.Qwen3VLModel._update_causal_mask = _update_causal_mask
    except (ImportError, AttributeError):
        pass


def print_trainable_parameters_visual(self):
    tb = [i for i,b in enumerate(self.blocks) if all(p.requires_grad for p in b.parameters())]
    nb = [i for i,b in enumerate(self.blocks) if not all(p.requires_grad for p in b.parameters())]
    print(f"Vision - Trainable blocks: {tb or 'None'}")
    print(f"Vision - Non-trainable blocks: {nb or 'None'}")
    print(f"Vision - Merger trainable: {any(p.requires_grad for p in self.merger.parameters())}")


def print_trainable_parameters(self):
    emb = getattr(self, "embed_tokens", None) or (hasattr(self,"model") and getattr(self.model,"embed_tokens",None))
    if emb: print(f"LLM - embed_tokens: {any(p.requires_grad for p in emb.parameters())}")
    layers = getattr(self,"layers",None) or (hasattr(self,"model") and getattr(self.model,"layers",None)) or []
    tl = [i for i,l in enumerate(layers) if any(p.requires_grad for p in l.parameters())]
    print(f"LLM - Trainable layers: {tl or 'None'}")


# ══════════════════════════════════════════════════════════════════════
# create_optimizer（含 ViT、Slot、StylePrefix 独立学习率）
# ══════════════════════════════════════════════════════════════════════

def create_optimizer(self):
    opt_model = self.model

    if self.optimizer is None:
        decay_params = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_params = [n for n in decay_params if "bias" not in n]

        projector_params = [n for n,_ in opt_model.named_parameters() if "merger" in n]
        slot_params      = [n for n,_ in opt_model.named_parameters() if "semantic_slot" in n]
        style_params     = [n for n,_ in opt_model.named_parameters() if "style_prefix" in n]
        # ★ ViT 参数（tune_mm_vision=True 时）
        vision_params    = [n for n,_ in opt_model.named_parameters()
                            if "visual.blocks" in n or "visual.patch_embed" in n
                            or "visual.pos_embed" in n or "visual.rotary_pos_emb" in n]

        special = set(projector_params + slot_params + style_params + vision_params)

        has_proj_lr    = bool(self.args.mm_projector_lr and self.args.mm_projector_lr != 0)
        has_vision_lr  = bool(self.args.vision_tower_lr and self.args.vision_tower_lr != 0)
        has_slot_lr    = bool(getattr(self.args,"slot_lr",None) and self.args.slot_lr != 0)
        has_style_lr   = bool(getattr(self.args,"style_lr",None) and self.args.style_lr != 0)

        # 主参数组（排除所有特殊参数）
        optimizer_grouped_parameters = [
            {"params": [p for n,p in opt_model.named_parameters()
                        if n in decay_params and n not in special and p.requires_grad],
             "weight_decay": self.args.weight_decay},
            {"params": [p for n,p in opt_model.named_parameters()
                        if n not in decay_params and n not in special and p.requires_grad],
             "weight_decay": 0.0},
        ]

        # Merger 参数组
        if has_proj_lr and projector_params:
            lr_p = self.args.mm_projector_lr
            optimizer_grouped_parameters += [
                {"params": [p for n,p in opt_model.named_parameters()
                            if n in decay_params and n in projector_params and p.requires_grad],
                 "weight_decay": self.args.weight_decay, "lr": lr_p},
                {"params": [p for n,p in opt_model.named_parameters()
                            if n not in decay_params and n in projector_params and p.requires_grad],
                 "weight_decay": 0.0, "lr": lr_p},
            ]
            print(f"[Optimizer] Merger  lr={lr_p:.2e}")

        # ★ ViT 参数组（tune_mm_vision=True 时）
        if has_vision_lr and vision_params:
            lr_v = self.args.vision_tower_lr
            optimizer_grouped_parameters += [
                {"params": [p for n,p in opt_model.named_parameters()
                            if n in vision_params and p.requires_grad],
                 "weight_decay": 0.0, "lr": lr_v},
            ]
            print(f"[Optimizer] ViT     lr={lr_v:.2e}")

        # SemanticSlot + CountHead 参数组
        if slot_params:
            lr_s = self.args.slot_lr if has_slot_lr else self.args.learning_rate
            optimizer_grouped_parameters += [
                {"params": [p for n,p in opt_model.named_parameters()
                            if n in decay_params and n in slot_params and p.requires_grad],
                 "weight_decay": self.args.weight_decay, "lr": lr_s},
                {"params": [p for n,p in opt_model.named_parameters()
                            if n not in decay_params and n in slot_params and p.requires_grad],
                 "weight_decay": 0.0, "lr": lr_s},
            ]
            print(f"[Optimizer] Slot    {len(slot_params)} params, lr={lr_s:.2e}")

        # StylePrefix 参数组（参数量极小，需要大 LR）
        if style_params:
            lr_st = self.args.style_lr if has_style_lr else self.args.learning_rate
            optimizer_grouped_parameters += [
                {"params": [p for n,p in opt_model.named_parameters()
                            if n in style_params and p.requires_grad],
                 "weight_decay": 0.0, "lr": lr_st},
            ]
            print(f"[Optimizer] Style   {len(style_params)} params, lr={lr_st:.2e}")

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if g["params"]]
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return self.optimizer


Trainer.create_optimizer = create_optimizer

TRAINABLE_KEYWORDS = [
    "merger", "patch_merger", "deepstack_merger",
    "lora_", "semantic_slot", "style_prefix",
]


def save_model(self, output_dir=None, _internal_call=False):
    if output_dir is None: output_dir = self.args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if self.args.local_rank not in (-1, 0): return
    if self.deepspeed: torch.cuda.synchronize()

    underlying = self.model
    if hasattr(self.model, "module"): underlying = self.model.module

    sd = {n: p.detach().cpu().clone()
          for n,p in underlying.named_parameters()
          if any(kw in n for kw in TRAINABLE_KEYWORDS)}

    if sd:
        sp = os.path.join(output_dir, "merger_weights.pt")
        torch.save(sd, sp)
        print(f"[Save] {len(sd)} tensors → {sp}")
        print(f"  Slot: {len([k for k in sd if 'semantic_slot' in k])}  "
              f"Style: {len([k for k in sd if 'style_prefix' in k])}")
    else:
        print("[Save][WARN] 无可训练参数")


Trainer.save_model = save_model

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
try:
    from transformers.trainer import TRAINER_STATE_NAME
except ImportError:
    TRAINER_STATE_NAME = "trainer_state.json"


def _save_checkpoint(self, model, trial, metrics=None):
    folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
    if self.hp_search_backend is None and trial is None: self.store_flos()
    run_dir = self._get_output_dir(trial=trial)
    out_dir = os.path.join(run_dir, folder)
    os.makedirs(out_dir, exist_ok=True)
    self.save_model(out_dir, _internal_call=True)
    if self.args.local_rank in (-1, 0):
        self.state.save_to_json(os.path.join(out_dir, TRAINER_STATE_NAME))
    if self.deepspeed:
        self.deepspeed.save_checkpoint(out_dir, exclude_frozen_parameters=True)
    if self.args.local_rank in (-1, 0) and self.args.save_total_limit is not None:
        import glob, shutil as _sh
        ckpts = sorted(glob.glob(os.path.join(run_dir, f"{PREFIX_CHECKPOINT_DIR}-*")),
                       key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0)
        for old in ckpts[:max(0, len(ckpts)-self.args.save_total_limit)]:
            _sh.rmtree(old, ignore_errors=True)
            print(f"[Checkpoint] Deleted: {old}")


Trainer._save_checkpoint = _save_checkpoint
Qwen3VLVisionModel.print_trainable_parameters = print_trainable_parameters_visual
Qwen3VLModel.print_trainable_parameters       = print_trainable_parameters