import os
from typing import Dict, List, Optional, Sequence

import datasets
import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from torch.utils.data import DataLoader, Sampler
from transformers import Trainer
from transformers.cache_utils import Cache

try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLVisionModel,
        Qwen3VLModel,
    )
except ImportError:
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VisionTransformerPretrainedModel as Qwen3VLVisionModel,
            Qwen3VLModel,
        )
    except ImportError:
        raise ImportError(
            "无法导入 Qwen3VLVisionModel / Qwen3VLModel，"
            "请升级：pip install 'transformers>=4.52.0'"
        )

try:
    from transformers.trainer import (
        ALL_LAYERNORM_LAYERS,
        get_parameter_names,
        has_length,
        is_sagemaker_mp_enabled,
    )
except ImportError:
    from transformers.trainer import (
        get_parameter_names,
        has_length,
    )
    is_sagemaker_mp_enabled = lambda: False
    ALL_LAYERNORM_LAYERS = (nn.LayerNorm, nn.RMSNorm)
from transformers.trainer_utils import seed_worker


def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    **kwargs,
):
    assert query_states.size(0) == key_states.size(0) == value_states.size(0) == 1
    query_states = query_states.squeeze(0)
    key_states   = key_states.squeeze(0)
    value_states = value_states.squeeze(0)
    cu_seqlens   = attention_mask

    with torch.no_grad():
        max_seqlen = max(
            [
                cu_seqlens[idx + 1] - cu_seqlens[idx]
                for idx in range(cu_seqlens.size(0) - 1)
            ]
        ).item()

    if not use_top_left_mask:
        causal = is_causal
    else:
        causal = is_causal and query_length != 1

    flash_kwargs = {}
    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    attn_output = flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=dropout,
        softmax_scale=softmax_scale,
        causal=causal,
        **flash_kwargs,
    )

    attn_output  = attn_output.unsqueeze(0)
    query_states = query_states.unsqueeze(0)
    key_states   = key_states.unsqueeze(0)
    value_states = value_states.unsqueeze(0)

    return attn_output


def _update_causal_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Cache,
    output_attentions: bool,
):
    return attention_mask


def replace_qwen2_vl_attention_class():
    """为 Qwen3-VL 注册 flatten attention patch"""
    import transformers

    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl
        modeling_qwen3_vl._flash_attention_forward = _flash_attention_forward
        modeling_qwen3_vl.Qwen3VLModel._update_causal_mask = _update_causal_mask
    except (ImportError, AttributeError):
        pass


# ── print_trainable_parameters ───────────────────────────────────────

def print_trainable_parameters_visual(self) -> None:
    trainable_blocks     = []
    non_trainable_blocks = []

    for block_idx, block in enumerate(self.blocks):
        is_trainable = all(param.requires_grad for param in block.parameters())
        if is_trainable:
            trainable_blocks.append(block_idx)
        else:
            non_trainable_blocks.append(block_idx)

    is_merger_trainable = any(param.requires_grad for param in self.merger.parameters())

    print("Vision Module - Attention Blocks:")
    print(f"Trainable Block Indices: {trainable_blocks if trainable_blocks else 'None'}")
    print(f"Non-Trainable Block Indices: {non_trainable_blocks if non_trainable_blocks else 'None'}")
    print(f"Merger Module Trainable: {is_merger_trainable}")


def print_trainable_parameters(self) -> None:
    if hasattr(self, 'embed_tokens'):
        embed_tokens = self.embed_tokens
    elif hasattr(self, 'model') and hasattr(self.model, 'embed_tokens'):
        embed_tokens = self.model.embed_tokens
    else:
        embed_tokens = None

    if embed_tokens is not None:
        is_embed_trainable = any(
            param.requires_grad for param in embed_tokens.parameters()
        )
        print(f"LLM Module - Embed Tokens Trainable: {is_embed_trainable}")
    else:
        print("LLM Module - Embed Tokens: Not found")

    trainable_layers     = []
    non_trainable_layers = []

    if hasattr(self, 'layers'):
        layers = self.layers
    elif hasattr(self, 'model') and hasattr(self.model, 'layers'):
        layers = self.model.layers
    else:
        layers = []

    for layer_idx, layer in enumerate(layers):
        is_trainable = any(param.requires_grad for param in layer.parameters())
        if is_trainable:
            trainable_layers.append(layer_idx)
        else:
            non_trainable_layers.append(layer_idx)

    print(f"LLM Module - Trainable Layer Indices: {trainable_layers if trainable_layers else 'None'}")
    print(f"LLM Module - Non-Trainable Layer Indices: {non_trainable_layers if non_trainable_layers else 'None'}")


# ── create_optimizer ─────────────────────────────────────────────────

def create_optimizer(self):
    opt_model = self.model

    if self.optimizer is None:
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        if self.args.mm_projector_lr is not None and self.args.mm_projector_lr != 0:
            projector_parameters = [
                name for name, _ in opt_model.named_parameters() if "merger" in name
            ]
            if self.args.vision_tower_lr is not None and self.args.vision_tower_lr != 0:
                vision_tower_parameters = [
                    name for name, _ in opt_model.named_parameters() if "visual" in name
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.vision_tower_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.vision_tower_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

        # 过滤掉空的参数组，避免 DeepSpeed IndexError
        optimizer_grouped_parameters = [
            g for g in optimizer_grouped_parameters if len(g["params"]) > 0
        ]

        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return self.optimizer


Trainer.create_optimizer = create_optimizer

# ── TRAINABLE_KEYWORDS ───────────────────────────────────────────────
TRAINABLE_KEYWORDS = ["merger", "patch_merger", "deepstack_merger", "lora_"]


# ── save_model ───────────────────────────────────────────────────────
def save_model(self, output_dir=None, _internal_call=False):
    if output_dir is None:
        output_dir = self.args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if self.args.local_rank not in (-1, 0):
        return

    if self.deepspeed:
        torch.cuda.synchronize()

    underlying_model = self.model
    if hasattr(self.model, 'module'):
        underlying_model = self.model.module

    trainable_state_dict = {}
    for n, p in underlying_model.named_parameters():
        if any(kw in n for kw in TRAINABLE_KEYWORDS):
            trainable_state_dict[n] = p.detach().cpu().clone()

    if trainable_state_dict:
        save_path = os.path.join(output_dir, "merger_weights.pt")
        torch.save(trainable_state_dict, save_path)
        print(f"[Save] {len(trainable_state_dict)} trainable tensors -> {save_path}")
    else:
        print(f"[Save][WARN] No trainable parameters found in {output_dir}")


Trainer.save_model = save_model


# ── _save_checkpoint ─────────────────────────────────────────────────
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
try:
    from transformers.trainer import TRAINER_STATE_NAME
except ImportError:
    TRAINER_STATE_NAME = "trainer_state.json"


def _save_checkpoint(self, model, trial, metrics=None):
    checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
    if self.hp_search_backend is None and trial is None:
        self.store_flos()

    run_dir    = self._get_output_dir(trial=trial)
    output_dir = os.path.join(run_dir, checkpoint_folder)
    os.makedirs(output_dir, exist_ok=True)

    self.save_model(output_dir, _internal_call=True)

    if self.args.local_rank in (-1, 0):
        self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

    if self.deepspeed:
        self.deepspeed.save_checkpoint(output_dir, exclude_frozen_parameters=True)

    if self.args.local_rank in (-1, 0) and self.args.save_total_limit is not None:
        import glob, shutil as _shutil
        _ckpts = sorted(
            glob.glob(os.path.join(run_dir, f"{PREFIX_CHECKPOINT_DIR}-*")),
            key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0,
        )
        if len(_ckpts) > self.args.save_total_limit:
            for _old in _ckpts[: len(_ckpts) - self.args.save_total_limit]:
                _shutil.rmtree(_old, ignore_errors=True)
                print(f"[Checkpoint] Deleted: {_old}")


Trainer._save_checkpoint = _save_checkpoint


# ── 注册 print_trainable_parameters ─────────────────────────────────
Qwen3VLVisionModel.print_trainable_parameters = print_trainable_parameters_visual
Qwen3VLModel.print_trainable_parameters       = print_trainable_parameters