"""
qwenvl/model/semantic_slot.py  (方案 D — 直接作为 LLM context tokens)

相对上一版 (B2 refine 版) 的改动:
  ★ 核心变化: Slot tokens 不再用来 refine 视觉, 而是直接作为
    独立的 9 个 context tokens 被 RSCaptionModel.forward 插入 LLM 输入序列.

  ★ 移除模块:
      - SlotRefineCrossAttention (视觉精炼, 贡献微弱)
      - refine_visual_tokens() API (替换成 get_slot_tokens)

  ★ 保留模块:
      - SemanticSlotAttention (聚合视觉 → 9 个 slot tokens)
      - SpatialPositionBias    (位置偏置, 可选)
      - CountAwareModule       (可选)

  ★ 新增 API:
      compute_slots(visual_tokens, grid_thw) → (B, K, D)
         计算 slot tokens, 由 RSCaptionModel.forward 调用并插入序列

  ★ 参数量变化:
      旧: SemanticSlotModule ≈ 335 M
      新: SemanticSlotModule ≈ 200 M  (去掉 slot_refine 的 135 M)

  ★ 梯度通路:
      旧: slot_init → slot_attn → [× refine_scale] × cross_attn → visual → LLM
          (乘法链, refine_scale ≈ 0.02 导致严重衰减)
      新: slot_init → slot_attn → directly into LLM input sequence → loss
          (直接通路, 梯度全强度)
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# SemanticSlotAttention (不变)
# ══════════════════════════════════════════════════════════════════════

class SemanticSlotAttention(nn.Module):
    def __init__(self, num_slots, dim, num_iterations=2, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_slots      = num_slots
        self.dim            = dim
        self.num_iterations = num_iterations
        self.scale          = dim ** -0.5

        self.slot_init  = nn.Parameter(torch.empty(num_slots, dim))
        nn.init.normal_(self.slot_init, std=0.02)

        self.norm_vis   = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.q_proj     = nn.Linear(dim, dim, bias=False)
        self.k_proj     = nn.Linear(dim, dim, bias=False)
        self.v_proj     = nn.Linear(dim, dim, bias=False)
        self.out_proj   = nn.Linear(dim, dim, bias=False)
        self.slot_sa    = nn.MultiheadAttention(dim, num_heads=num_heads,
                                                dropout=dropout, batch_first=True)
        self.norm_sa    = nn.LayerNorm(dim)
        self.norm_ffn   = nn.LayerNorm(dim)
        self.ffn        = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )
        self._last_attn_weights: Optional[torch.Tensor] = None

    def forward(self, visual_tokens, spatial_bias=None):
        B  = visual_tokens.shape[0]
        vn = self.norm_vis(visual_tokens)
        slots = self.slot_init.unsqueeze(0).expand(B, -1, -1).contiguous()

        for _ in range(self.num_iterations):
            sn  = self.norm_slots(slots)
            q, k, v = self.q_proj(sn), self.k_proj(vn), self.v_proj(vn)
            attn = torch.einsum("bkd,bnd->bkn", q, k) * self.scale
            if spatial_bias is not None:
                attn = attn + spatial_bias.to(attn.dtype)
            aw   = attn.softmax(dim=1)
            self._last_attn_weights = aw.detach()
            slots = slots + self.out_proj(torch.einsum("bkn,bnd->bkd", aw, v))
            sa, _ = self.slot_sa(slots, slots, slots)
            slots = self.norm_sa(slots + sa)

        return slots + self.ffn(self.norm_ffn(slots))


# ══════════════════════════════════════════════════════════════════════
# CountAwareModule (不变, 可选)
# ══════════════════════════════════════════════════════════════════════

class CountAwareModule(nn.Module):
    NUM_CLASSES = 6

    def __init__(self, dim: int):
        super().__init__()
        hidden = max(dim // 16, 64)
        self.token_classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.NUM_CLASSES),
        )
        self.count_proj  = nn.Linear(self.NUM_CLASSES + 1, dim)
        self.count_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, visual_tokens, attn_weights):
        token_probs = self.token_classifier(visual_tokens).softmax(dim=-1)
        aw_norm     = attn_weights / (attn_weights.sum(2, keepdim=True) + 1e-9)
        slot_class  = torch.einsum("bkn,bnc->bkc", aw_norm, token_probs)
        density     = torch.log1p(
            attn_weights.sum(2, keepdim=True) * visual_tokens.shape[1]
        )
        signal = torch.cat([slot_class, density], dim=-1)
        return self.count_scale * self.count_proj(signal)


# ══════════════════════════════════════════════════════════════════════
# SpatialPositionBias (不变)
# ══════════════════════════════════════════════════════════════════════

class SpatialPositionBias(nn.Module):
    def __init__(self, num_slots, init_gamma=4.0):
        super().__init__()
        self.num_slots = num_slots
        self.log_gamma = nn.Parameter(torch.tensor(math.log(init_gamma)))
        grid_rows = max(1, int(num_slots ** 0.5))
        grid_cols = math.ceil(num_slots / grid_rows)
        hs = torch.linspace(0, 1, grid_rows)
        ws = torch.linspace(0, 1, grid_cols)
        gh, gw = torch.meshgrid(hs, ws, indexing="ij")
        self.register_buffer(
            "slot_coords",
            torch.stack([gh.flatten(), gw.flatten()], dim=-1)[:num_slots],
        )

    def forward(self, H_m: int, W_m: int, device=None) -> torch.Tensor:
        if device is None:
            device = self.slot_coords.device
        tok_h = torch.linspace(0, 1, H_m, device=device)
        tok_w = torch.linspace(0, 1, W_m, device=device)
        gh, gw = torch.meshgrid(tok_h, tok_w, indexing="ij")
        tc   = torch.stack([gh.flatten(), gw.flatten()], dim=-1)
        sc   = self.slot_coords.to(device).unsqueeze(1)
        dist = ((sc - tc.unsqueeze(0)) ** 2).sum(-1)
        return (-torch.exp(self.log_gamma) * dist).unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════
# SemanticSlotModule — 方案 D 版本
# ══════════════════════════════════════════════════════════════════════

class SemanticSlotModule(nn.Module):
    """
    方案 D: Slot tokens 作为 LLM context 直接插入序列.

    Forward 流程:
        RSCaptionModel.forward 调用 compute_slots(visual_tokens, grid_thw)
          → 得到 (B, K, D) 的 slot tokens
          → 拼入 LLM 输入序列 [vision_end, SLOT×K, STYLE×K2, text]

    梯度通路:
        loss → LLM → slot_tokens → compute_slots → slot_attn → slot_init ✓
        完全直接, 无 refine_scale / gate 衰减.
    """

    def __init__(
        self,
        num_slots         = 9,
        dim               = 4096,
        num_iterations    = 2,
        use_position_info = True,
        merge_size        = 2,
        num_heads         = 8,
        use_count_head    = False,
    ):
        super().__init__()
        self.num_slots         = num_slots
        self.dim               = dim
        self.use_position_info = use_position_info
        self.merge_size        = merge_size

        self.slot_attn   = SemanticSlotAttention(num_slots, dim, num_iterations, num_heads)
        self.spatial_bias_module = (
            SpatialPositionBias(num_slots) if use_position_info else None
        )
        self.count_head  = CountAwareModule(dim) if use_count_head else None

        # ★ 移除 slot_refine (SlotRefineCrossAttention), 省 135 M 参数

        # 首批 grid 信息只打印一次, 诊断用
        self._grid_logged = False

    # ──────────────────────────────────────────────────────────────────
    # 保留接口 (no-op, 向后兼容)
    # ──────────────────────────────────────────────────────────────────

    def register_merger_hook(self, model: nn.Module):
        """方案 D 下无需 hook, 保留签名为向后兼容"""
        logger.info(
            f"[SemanticSlot] 方案 D 模式 — slot 作为 LLM context token 直接插入, "
            f"通过 RSCaptionModel.forward 调 compute_slots()"
        )

    def remove_hooks(self):
        pass

    # ──────────────────────────────────────────────────────────────────
    # Grid 推算 (不变)
    # ──────────────────────────────────────────────────────────────────

    def _get_grid_info(
        self, N_total: int, grid_thw: Optional[torch.Tensor]
    ) -> Tuple[int, int, int, int, bool]:
        if grid_thw is not None and len(grid_thw) > 0:
            first_thw = grid_thw[0]
            is_homogeneous = True
            for i in range(1, len(grid_thw)):
                if (int(grid_thw[i][0]) != int(first_thw[0])
                        or int(grid_thw[i][1]) != int(first_thw[1])
                        or int(grid_thw[i][2]) != int(first_thw[2])):
                    is_homogeneous = False
                    break

            if is_homogeneous:
                t_val = int(first_thw[0])
                h_val = int(first_thw[1])
                w_val = int(first_thw[2])
                H_m = max(h_val // self.merge_size, 1)
                W_m = max(w_val // self.merge_size, 1)
                N_per = t_val * H_m * W_m
                B = len(grid_thw)

                if N_per * B == N_total:
                    return B, N_per, H_m, W_m, True

            total_calc = 0
            for i in range(len(grid_thw)):
                t = int(grid_thw[i][0])
                h = int(grid_thw[i][1])
                w = int(grid_thw[i][2])
                total_calc += t * (h // self.merge_size) * (w // self.merge_size)
            if total_calc == N_total:
                side = max(1, int(math.sqrt(N_total)))
                return 1, N_total, side, side, False

        side = max(1, int(math.sqrt(N_total)))
        ok = (side * side == N_total)
        return 1, N_total, side, side, ok

    # ──────────────────────────────────────────────────────────────────
    # ★ 方案 D 核心 API: compute_slots
    # ──────────────────────────────────────────────────────────────────

    def compute_slots(
        self,
        visual_tokens: torch.Tensor,          # (N_total, D) post-merger
        grid_thw:      Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        从 post-merger 视觉 tokens 聚合出 (B, K, D) 的 slot tokens.

        梯度路径: slot_init → slot_attn → slots → 返回 → 插入 LLM 序列 → loss

        Args:
          visual_tokens: (N_total, D)  post-merger, N_total = B × N_per
          grid_thw:      (num_images, 3)  用于推算 B, H_m, W_m

        Returns:
          slots: (B, K, D)  每个样本独立的 K 个 slot tokens
        """
        if visual_tokens.dim() != 2:
            logger.warning(
                f"[SemanticSlot] compute_slots 输入 shape {visual_tokens.shape} "
                f"不是 (N_total, D), 无法处理"
            )
            return None

        N_total, D = visual_tokens.shape
        device, dtype = visual_tokens.device, visual_tokens.dtype

        B, N_per, H_m, W_m, grid_ok = self._get_grid_info(N_total, grid_thw)

        # 首批诊断输出
        if not self._grid_logged:
            logger.info(
                f"[SemanticSlot] 首次 compute_slots:\n"
                f"  N_total={N_total}  D={D}\n"
                f"  grid_thw={grid_thw.tolist() if grid_thw is not None else None}\n"
                f"  推算: B={B}  N_per={N_per}  H_m={H_m}  W_m={W_m}  ok={grid_ok}"
            )
            self._grid_logged = True

        try:
            # ★ 关键: 按 batch 维度 reshape
            #   旧版是 vis = visual_tokens.unsqueeze(0), 把整个 batch 当 1 个样本.
            #   这会让所有样本共享 slots, style_prefix 读的 _last_slots 也是共享的.
            #   新版要保持 per-sample slots, 因为要插到每个样本独立的序列里.
            if grid_ok and B > 1:
                # (B * N_per, D) → (B, N_per, D)
                vis = visual_tokens.view(B, N_per, D)
            else:
                # batch=1 或异构, 保持单批
                vis = visual_tokens.unsqueeze(0)
                B = 1

            # spatial bias (可选)
            spatial_bias = None
            if (self.use_position_info
                    and self.spatial_bias_module is not None
                    and grid_ok):
                per_img_bias = self.spatial_bias_module(H_m, W_m, device).to(dtype=dtype)
                # per_img_bias: (1, K, H_m*W_m)
                # 对每个样本都用同一张 bias (因为每个样本空间结构相同)
                spatial_bias = per_img_bias.expand(B, -1, -1)
                if spatial_bias.shape[-1] != N_per:
                    logger.warning(
                        f"[SemanticSlot] spatial_bias shape[-1]={spatial_bias.shape[-1]} "
                        f"!= N_per={N_per}, 禁用 spatial_bias"
                    )
                    spatial_bias = None

            slots = self.slot_attn(vis, spatial_bias)  # (B, K, D)

            # count embedding (可选)
            if (self.count_head is not None
                    and self.slot_attn._last_attn_weights is not None):
                aw          = self.slot_attn._last_attn_weights.to(dtype=dtype)
                count_embed = self.count_head(vis, aw)
                slots       = slots + count_embed

            return slots   # (B, K, D)

        except Exception as e:
            import traceback
            logger.error(
                f"[SemanticSlot] compute_slots 失败: {e}\n{traceback.format_exc()}"
            )
            return None

    # ──────────────────────────────────────────────────────────────────
    # 调试
    # ──────────────────────────────────────────────────────────────────

    def print_params(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[SemanticSlot] 参数量: {total/1e6:.2f}M "
            f"(trainable={trainable/1e6:.2f}M)\n"
            f"  num_slots      : {self.num_slots}\n"
            f"  use_position   : {self.use_position_info}\n"
            f"  count_head     : {'enabled' if self.count_head else 'disabled'}\n"
            f"  ★ 方案 D: 无 slot_refine, slot 作为 LLM context token 直接起作用"
        )