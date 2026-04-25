"""
qwenvl/model/rs_caption_model.py  (方案 D — slot+style 直接作为 LLM context)

相对 B2 版本的改动:
  ★ 新增 _inject_slot_and_style_tokens
    在 vision_end 之后同时插入 [slot_tokens × K_slot] 和 [style_tokens × K_style]
    选项 1 排列: [vision_end] [SLOT×9] [STYLE×8] [text...]

  ★ _process_visual 不再调 slot.refine_visual_tokens
    保留原始 post-merger 视觉 tokens, slot 通过 compute_slots 另行生成.

  ★ _compute_extended_position 扩展支持两段 token
    Slot 时间轴连续递增, 继承视觉的空间 (h, w) 编码
    Style 时间轴继续递增, 空间轴使用独立的单轴递增 (不受空间锚定)

  ★ 注入条件保留: 仅当 input_ids 有 VISION_END_ID 时才插入
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import Qwen3VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__name__)

VISION_END_ID = 151653
IGNORE_INDEX  = -100


class RSCaptionModel(Qwen3VLForConditionalGeneration):
    """遥感图像描述专用模型 (方案 D)."""

    def __init__(self, config):
        super().__init__(config)
        self.semantic_slot: Optional[nn.Module] = None
        self.style_prefix:  Optional[nn.Module] = None

    # ──────────────────────────────────────────────────────────────────
    # 模块挂载
    # ──────────────────────────────────────────────────────────────────

    def attach_modules(
        self,
        slot_module:  Optional[nn.Module] = None,
        style_module: Optional[nn.Module] = None,
    ):
        """挂载 SemanticSlot / StylePrefix. 方案 D 下 slot-style 独立, 不需要互相引用."""
        if slot_module is not None:
            self.semantic_slot = slot_module
            if hasattr(slot_module, "register_merger_hook"):
                slot_module.register_merger_hook(self)  # 方案 D 下是 no-op
            logger.info(
                f"[RSCaptionModel] SemanticSlot 挂载 (K={slot_module.num_slots}) "
                f"[方案 D: 作为 LLM context token]"
            )

        if style_module is not None:
            self.style_prefix = style_module
            # ★ 方案 D: style 不再引用 slot
            # 但为了兼容旧代码, 仍调 set_slot_module (style_prefix.py 里是 no-op)
            if slot_module is not None and hasattr(style_module, "set_slot_module"):
                style_module.set_slot_module(slot_module)
            logger.info(
                f"[RSCaptionModel] StylePrefix 挂载 "
                f"(K={style_module.num_style_tokens}) [方案 D]"
            )

    # ──────────────────────────────────────────────────────────────────
    # visual 输出提取 (不变)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_post_merger_embed(visual_out) -> torch.Tensor:
        """从 Qwen3VL visual 输出中提取 pooler_output (post-merger tokens)."""
        if isinstance(visual_out, torch.Tensor):
            return visual_out

        pooler = getattr(visual_out, "pooler_output", None)
        if pooler is not None and isinstance(pooler, torch.Tensor):
            return pooler

        last_hidden = getattr(visual_out, "last_hidden_state", None)
        if last_hidden is not None:
            logger.warning(
                "[RSCaptionModel] visual 返回无 pooler_output, 回退 last_hidden_state"
            )
            return last_hidden

        if isinstance(visual_out, (tuple, list)) and len(visual_out) > 0:
            first = visual_out[0]
            if isinstance(first, torch.Tensor):
                return first

        raise TypeError(
            f"无法从 visual 输出提取 embedding: type={type(visual_out).__name__}"
        )

    def _process_visual_and_slots(
        self,
        pixel_values:   torch.Tensor,
        grid_thw:       torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        运行 visual → 返回 (post-merger visual tokens, slot_tokens 或 None).

        ★ 方案 D 改动:
          - visual tokens 不再被 slot refine 修改, 直接返回原始 pooler_output
          - 另外计算 slot_tokens, 供后续插入 LLM 序列

        Returns:
          visual_embeds: (N_total, D)  post-merger, 原样用于 masked_scatter
          slot_tokens:   (B, K, D) 或 None
        """
        vis_dtype    = next(self.model.visual.parameters()).dtype
        pixel_values = pixel_values.type(vis_dtype)
        visual_out   = self.model.visual(pixel_values, grid_thw=grid_thw)
        embeds       = self._extract_post_merger_embed(visual_out)  # (N_total, D)

        slot_tokens = None
        if self.semantic_slot is not None:
            slot_tokens = self.semantic_slot.compute_slots(
                visual_tokens = embeds,
                grid_thw      = grid_thw,
            )

        return embeds, slot_tokens

    # ──────────────────────────────────────────────────────────────────
    # forward (重写)
    # ──────────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:              Optional[torch.LongTensor]   = None,
        attention_mask:         Optional[torch.Tensor]       = None,
        position_ids:           Optional[torch.LongTensor]   = None,
        past_key_values:        Optional[List]               = None,
        inputs_embeds:          Optional[torch.FloatTensor]  = None,
        labels:                 Optional[torch.LongTensor]   = None,
        use_cache:              Optional[bool]               = None,
        output_attentions:      Optional[bool]               = None,
        output_hidden_states:   Optional[bool]               = None,
        return_dict:            Optional[bool]               = None,
        pixel_values:           Optional[torch.Tensor]       = None,
        pixel_values_videos:    Optional[torch.FloatTensor]  = None,
        image_grid_thw:         Optional[torch.LongTensor]   = None,
        video_grid_thw:         Optional[torch.LongTensor]   = None,
        **kwargs,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # ══════════════════════════════════════════════════════════════
        # 阶段 1: 构建 inputs_embeds + 预计算 slot_tokens
        # ══════════════════════════════════════════════════════════════

        slot_tokens_for_inject: Optional[torch.Tensor] = None

        if inputs_embeds is None:
            inputs_embeds = self.model.language_model.embed_tokens(input_ids)

            if pixel_values is not None:
                image_embeds, slot_tokens_for_inject = self._process_visual_and_slots(
                    pixel_values, image_grid_thw
                )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                video_embeds, video_slot_tokens = self._process_visual_and_slots(
                    pixel_values_videos, video_grid_thw
                )
                # 视频场景: 如果同一个 batch 同时有图有视频, slot 以 image 的为准
                if slot_tokens_for_inject is None:
                    slot_tokens_for_inject = video_slot_tokens

                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # ══════════════════════════════════════════════════════════════
        # 阶段 2: 注入 Slot + Style Tokens (方案 D 核心)
        # ══════════════════════════════════════════════════════════════

        # 条件: 有 style 或 slot 模块 + prefill 步 + 有 VISION_END_ID
        has_slot  = self.semantic_slot is not None and slot_tokens_for_inject is not None
        has_style = self.style_prefix is not None

        should_inject = (
            (has_slot or has_style)
            and input_ids is not None
            and input_ids.shape[1] > 1
            and (input_ids == VISION_END_ID).any()
            and (
                position_ids is None
                or (position_ids.dim() == 3 and position_ids.shape[0] == 3)
            )
        )

        if should_inject:
            inputs_embeds, attention_mask, position_ids, labels, added_len = \
                self._inject_slot_and_style_tokens(
                    input_ids     = input_ids,
                    inputs_embeds = inputs_embeds,
                    attention_mask= attention_mask,
                    position_ids  = position_ids,
                    labels        = labels,
                    slot_tokens   = slot_tokens_for_inject,   # (B, K_slot, D) or None
                )

            # 对齐 input_ids 长度 (用 pad_token_id 填充插入位置)
            B_in, orig_seq_in = input_ids.shape
            _text_cfg = getattr(self.config, "text_config", self.config)
            pad_id    = getattr(_text_cfg, "pad_token_id", None) or 0

            padded_input_ids = torch.full(
                (B_in, orig_seq_in + added_len), pad_id,
                device=input_ids.device, dtype=input_ids.dtype,
            )
            for b in range(B_in):
                idx = (input_ids[b] == VISION_END_ID).nonzero(as_tuple=True)[0]
                pos = idx[-1].item() + 1 if len(idx) > 0 else orig_seq_in
                padded_input_ids[b, :pos]             = input_ids[b, :pos]
                padded_input_ids[b, pos + added_len:] = input_ids[b, pos:]
            input_ids = padded_input_ids

        # ══════════════════════════════════════════════════════════════
        # 阶段 3: 送入 language model (不变)
        # ══════════════════════════════════════════════════════════════

        lm_input_ids = None if inputs_embeds is not None else input_ids

        outputs = self.model.language_model(
            input_ids            = lm_input_ids,
            attention_mask       = attention_mask,
            position_ids         = position_ids,
            past_key_values      = past_key_values,
            inputs_embeds        = inputs_embeds,
            use_cache            = use_cache,
            output_attentions    = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict          = True,
        )

        hidden_states = outputs[0]
        logits        = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        if not return_dict:
            out = (logits,) + outputs[1:]
            return (loss,) + out if loss is not None else out

        return CausalLMOutputWithPast(
            loss             = loss,
            logits           = logits,
            past_key_values  = outputs.past_key_values,
            hidden_states    = outputs.hidden_states,
            attentions       = outputs.attentions,
        )

    # ──────────────────────────────────────────────────────────────────
    # Slot + Style Token 注入 (方案 D 核心)
    # ──────────────────────────────────────────────────────────────────

    def _inject_slot_and_style_tokens(
        self,
        input_ids:      torch.LongTensor,
        inputs_embeds:  torch.FloatTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids:   Optional[torch.LongTensor],
        labels:         Optional[torch.LongTensor],
        slot_tokens:    Optional[torch.Tensor],  # (B, K_slot, D) or None
    ) -> Tuple[
        torch.Tensor,              # new_embeds
        Optional[torch.Tensor],    # new_mask
        Optional[torch.Tensor],    # new_pos
        Optional[torch.Tensor],    # new_labels
        int,                       # added_len (K_slot + K_style)
    ]:
        """
        在每个样本的 VISION_END 之后插入 [slot_tokens, style_tokens].

        选项 1 排列: [vision_end] [SLOT × K_slot] [STYLE × K_style] [text...]
        """
        B, orig_seq = input_ids.shape
        D = inputs_embeds.size(-1)
        device, dtype = inputs_embeds.device, inputs_embeds.dtype

        # 确定 K_slot / K_style
        K_slot  = slot_tokens.size(1) if slot_tokens is not None else 0
        K_style = self.style_prefix.num_style_tokens if self.style_prefix is not None else 0
        K_total = K_slot + K_style

        if K_total == 0:
            return inputs_embeds, attention_mask, position_ids, labels, 0

        # 生成 style_tokens
        if K_style > 0:
            style_tokens = self.style_prefix.get_style_tokens(B, device, dtype)  # (B, K_style, D)
        else:
            style_tokens = None

        # 处理 slot_tokens 的 batch 对齐
        # compute_slots 返回的可能是 (B, K, D) 也可能是 (1, K, D) (异构场景)
        if slot_tokens is not None:
            if slot_tokens.size(0) == 1 and B > 1:
                slot_tokens = slot_tokens.expand(B, -1, -1)
            slot_tokens = slot_tokens.to(device=device, dtype=dtype)

        # 定位每个样本的插入位置 (VISION_END_ID 之后)
        insert_positions: List[Optional[int]] = []
        for b in range(B):
            idx = (input_ids[b] == VISION_END_ID).nonzero(as_tuple=True)[0]
            insert_positions.append(idx[-1].item() + 1 if len(idx) > 0 else None)

        new_len = orig_seq + K_total

        # 预分配输出
        new_embeds = torch.zeros(B, new_len, D, device=device, dtype=dtype)

        new_mask = None
        if attention_mask is not None:
            new_mask = torch.zeros(B, new_len, device=device, dtype=attention_mask.dtype)

        new_labels = None
        if labels is not None:
            new_labels = torch.full(
                (B, new_len), IGNORE_INDEX, device=device, dtype=labels.dtype
            )

        new_pos = None
        if position_ids is not None:
            if position_ids.dim() == 3 and position_ids.shape[0] == 3:
                new_pos = torch.zeros(3, B, new_len, device=device, dtype=position_ids.dtype)
            else:
                logger.debug(
                    f"[inject] position_ids shape {position_ids.shape} 非 M-RoPE 格式, 跳过扩展"
                )
                position_ids = None

        for b in range(B):
            pos = insert_positions[b]

            # 构造 [slot_b][style_b] 拼接 token (K_total, D)
            pieces = []
            if slot_tokens is not None:
                pieces.append(slot_tokens[b])  # (K_slot, D)
            if style_tokens is not None:
                pieces.append(style_tokens[b])  # (K_style, D)
            injected = torch.cat(pieces, dim=0)  # (K_total, D)

            if pos is None:
                # 无 VISION_END: 追加到末尾
                new_embeds[b, :orig_seq] = inputs_embeds[b]
                new_embeds[b, orig_seq:] = injected
                if new_mask is not None:
                    new_mask[b, :orig_seq] = attention_mask[b]
                if new_labels is not None:
                    new_labels[b, :orig_seq] = labels[b]
                if new_pos is not None:
                    new_pos[:, b, :orig_seq] = position_ids[:, b, :]
                    last = position_ids[:, b, -1:]
                    pad  = torch.arange(1, K_total + 1, device=device, dtype=position_ids.dtype)
                    new_pos[:, b, orig_seq:] = last + pad.unsqueeze(0)
                continue

            # 正常插入
            new_embeds[b, :pos]              = inputs_embeds[b, :pos]
            new_embeds[b, pos:pos+K_total]   = injected
            new_embeds[b, pos+K_total:]      = inputs_embeds[b, pos:]

            if new_mask is not None:
                new_mask[b, :pos]              = attention_mask[b, :pos]
                new_mask[b, pos:pos+K_total]   = 1
                new_mask[b, pos+K_total:]      = attention_mask[b, pos:]

            if new_labels is not None:
                new_labels[b, :pos]              = labels[b, :pos]
                new_labels[b, pos+K_total:]      = labels[b, pos:]

            if new_pos is not None:
                new_pos[:, b, :] = self._compute_extended_position(
                    position_ids[:, b, :], pos, K_slot, K_style
                )

        return new_embeds, new_mask, new_pos, new_labels, K_total

    # ──────────────────────────────────────────────────────────────────
    # Position 扩展 (方案 D — 支持两段)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_extended_position(
        pid: torch.Tensor,   # (3, seq)
        pos: int,
        K_slot:  int,
        K_style: int,
    ) -> torch.Tensor:
        """
        扩展 position_ids, 在 pos 位置插入 K_slot + K_style 个 token 的 position.

        策略:
          - Slot 的时间轴: 紧接 vision_end 连续递增
            Slot 的空间轴 (h, w): 继承 vision_end 的 h/w (slot 在空间上锚定视觉区域)
          - Style 的时间轴: 继续连续递增
            Style 的空间轴 (h, w): 使用独立单轴递增 (style 不需要空间锚定)
          - 后续文本的时间轴: 整体 +K_total
        """
        device = pid.device
        dtype  = pid.dtype
        K_total = K_slot + K_style

        prev = pid[:, pos - 1]  # (3,), vision_end 的 position

        # === Slot 段 ===
        if K_slot > 0:
            slot_t = torch.arange(
                int(prev[0].item()) + 1,
                int(prev[0].item()) + 1 + K_slot,
                device=device, dtype=dtype,
            )
            # Slot 空间轴: 继承 vision_end (slot 在视觉空间上)
            slot_h = prev[1].expand(K_slot)
            slot_w = prev[2].expand(K_slot)
            slot_pos = torch.stack([slot_t, slot_h, slot_w], dim=0)  # (3, K_slot)
        else:
            slot_pos = None

        # === Style 段 ===
        if K_style > 0:
            style_t_start = int(prev[0].item()) + 1 + K_slot
            style_t = torch.arange(
                style_t_start,
                style_t_start + K_style,
                device=device, dtype=dtype,
            )
            # Style 空间轴: 使用与时间轴同步的递增 (style 无空间语义, 避免三轴全相同导致 RoPE 退化)
            style_h = torch.arange(
                int(prev[1].item()) + K_slot + 1,
                int(prev[1].item()) + K_slot + 1 + K_style,
                device=device, dtype=dtype,
            )
            style_w = torch.arange(
                int(prev[2].item()) + K_slot + 1,
                int(prev[2].item()) + K_slot + 1 + K_style,
                device=device, dtype=dtype,
            )
            style_pos = torch.stack([style_t, style_h, style_w], dim=0)  # (3, K_style)
        else:
            style_pos = None

        # 后续文本: 时间轴 +K_total, 空间轴不变
        tail = pid[:, pos:].clone()
        tail[0] = tail[0] + K_total

        # 拼接
        pieces = [pid[:, :pos]]
        if slot_pos is not None:
            pieces.append(slot_pos)
        if style_pos is not None:
            pieces.append(style_pos)
        pieces.append(tail)

        return torch.cat(pieces, dim=1)  # (3, orig_seq + K_total)

    # ──────────────────────────────────────────────────────────────────
    # generate 兼容
    # ──────────────────────────────────────────────────────────────────

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return super().prepare_inputs_for_generation(input_ids, **kwargs)