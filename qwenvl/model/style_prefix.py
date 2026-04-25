"""
qwenvl/model/style_prefix.py  (方案 D — 独立 style tokens, 不吸收 slot)

相对上一版的改动:
  ★ 移除 slot_cross_attn / slot_cross_norm / slot_gate
    诊断显示 slot_gate ≈ 0, 这条通路从未真正工作.
    方案 D 下, slot 作为独立的 LLM context token 直接起作用,
    style 不再需要"吸收 slot 信息".

  ★ 保留核心功能:
    - style_embeds: K 个可学习 token
    - initialize_from_sentences: 从典型句子初始化 (句号 bug 已修)
    - get_style_tokens(B, device, dtype) API

  ★ 参数量变化:
    旧: 约 67 M (含 slot_cross_attn 的 4 套 Linear + LayerNorm + gate)
    新: 约 33 M (只有 style_embeds 本身)
"""

import logging
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class StylePrefixModule(nn.Module):
    """
    方案 D: 独立 style prefix tokens.

    K 个可学习的 style embedding, 通过 RSCaptionModel.forward 被插入 LLM
    输入序列的 vision_end 之后 (紧邻 slot tokens).

    Slot-style 不再有直接连接 — 各自独立作为 LLM context.
    """

    def __init__(
        self,
        num_style_tokens: int = 8,
        dim:              int = 4096,
        use_slot_attention: bool = False,   # ★ 方案 D 下保留参数以兼容旧调用, 但不用
        num_heads:        int = 8,          # ★ 保留以兼容旧调用
    ):
        super().__init__()
        self.num_style_tokens = num_style_tokens
        self.dim              = dim
        # ★ 方案 D 下 use_slot_attention 无效, 记录但不使用
        self.use_slot_attention = False
        if use_slot_attention:
            logger.info(
                "[StylePrefix] 方案 D: use_slot_attention=True 被忽略, "
                "style 不再直接吸收 slot. slot 现在作为独立 LLM context token."
            )

        self.style_embeds = nn.Parameter(torch.empty(num_style_tokens, dim))
        nn.init.normal_(self.style_embeds, std=0.02)

    # ──────────────────────────────────────────────────────────────────
    # 兼容旧接口 (no-op)
    # ──────────────────────────────────────────────────────────────────

    def set_slot_module(self, slot_module: nn.Module):
        """方案 D 下 style 不需要直接知道 slot 模块, 保留以兼容旧调用链"""
        logger.debug("[StylePrefix] 方案 D: set_slot_module 被调用但无作用 (兼容)")

    # ──────────────────────────────────────────────────────────────────
    # 核心 API: 构造 style tokens
    # ──────────────────────────────────────────────────────────────────

    def get_style_tokens(
        self, B: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        返回 (B, K, D) 的 style tokens, 梯度流通到 style_embeds.

        方案 D: 不再和 slot 交互, 直接 expand 给 batch.
        """
        st      = self.style_embeds.to(device=device, dtype=dtype)
        style_b = st.unsqueeze(0).expand(B, -1, -1).contiguous()
        return style_b

    # ──────────────────────────────────────────────────────────────────
    # 从句子初始化 (保留)
    # ──────────────────────────────────────────────────────────────────

    def initialize_from_sentences(
        self,
        sentences: List[str],
        tokenizer,
        embed_fn,
    ):
        """
        从典型句子初始化 style_embeds.

        策略:
          1. rstrip 去除末尾标点 (避免 embeds[-1] 取到 '.' 的向量)
          2. 末尾 3 个 token 的 mean (语义集中 + 稳定)
          3. 叠加小噪声 (避免 K 个 slot 初始化过于相似)
        """
        with torch.no_grad():
            try:
                target_device = next(embed_fn.parameters()).device
            except (StopIteration, AttributeError):
                target_device = self.style_embeds.device

            n = min(len(sentences), self.num_style_tokens)
            for i, sent in enumerate(sentences[:n]):
                sent_clean = sent.rstrip(" \t\n.。!?!?,,;;:")
                if not sent_clean:
                    continue

                ids = tokenizer.encode(
                    sent_clean,
                    return_tensors    = "pt",
                    add_special_tokens = False,
                ).to(target_device)
                embeds = embed_fn(ids).squeeze(0)

                tail_k    = min(3, embeds.size(0))
                tail_mean = embeds[-tail_k:].mean(dim=0)

                self.style_embeds.data[i] = tail_mean.to(
                    device=self.style_embeds.device,
                    dtype =self.style_embeds.dtype,
                )

            if self.num_style_tokens > n:
                nn.init.normal_(self.style_embeds.data[n:], std=0.02)

            self.style_embeds.data += torch.randn_like(
                self.style_embeds.data
            ) * 0.01

        logger.info(
            f"[StylePrefix] 从 {n}/{self.num_style_tokens} 个句子初始化完成"
        )

    # ──────────────────────────────────────────────────────────────────
    # 调试
    # ──────────────────────────────────────────────────────────────────

    def print_params(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[StylePrefix] 参数量: {total/1e6:.4f}M "
            f"(trainable={trainable/1e6:.4f}M)\n"
            f"  style_embeds : {list(self.style_embeds.shape)}\n"
            f"  ★ 方案 D: 独立 style tokens, 不与 slot 交互"
        )