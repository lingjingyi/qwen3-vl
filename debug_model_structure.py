#!/usr/bin/env python3
"""
debug_model_structure.py

用途：
  1. 打印 Qwen3-VL 完整模块树，确认 SemanticSlotToken 的 hook 插入位置
  2. 追踪一次前向传播中各关键节点的 tensor shape
  3. 确认视觉 token 和文本 token 在 LLM 输入侧的拼接位置
  4. 打印 position_ids / attention_mask 的结构，为 hook 修改提供依据

用法：
  python /opt/data/private/qwen3-vl-master/qwen3-vl/debug_model_structure.py \
      --model_path /opt/data/private/qwen3-vl-master/qwen3-vl/pretrained/Qwen3-VL-8B-Instruct \
      --image_path /opt/data/private/qwen3-vl-master/data/university-s-train/0013.jpg \
      [--verbose_tensor]   # 打印每层的输入输出shape（信息量大）
      [--find_hook_point]  # 专门追踪视觉token的流向
"""

#!/usr/bin/env python3
"""
debug_model_structure.py（修复版）

修复内容（对应输出中的三个错误）：
  1. find_key_modules：改用直接属性访问，正确定位 language_model.layers
  2. 前向传播：修复 "not enough values to unpack" 错误，分阶段追踪
  3. Hook 注册：确保注册在正确的 LLM layer 上，而非 ViT block

新增：
  --use_position_info 开关（为 SemanticSlotToken 的位置编码分支准备）

用法：
  python debug_model_structure.py \
      --model_path /opt/data/private/qwen3-vl-master/qwen3-vl/pretrained/Qwen3-VL-8B-Instruct \
      --image_path /opt/data/private/qwen3-vl-master/data/university-s-train/0013.jpg \
      --use_position_info \
      --output_json /tmp/model_structure.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SEP = "═" * 70


# ══════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════

def shape_str(t: Any) -> str:
    if t is None:
        return "None"
    if isinstance(t, torch.Tensor):
        return f"Tensor{list(t.shape)} dtype={t.dtype} device={t.device}"
    if isinstance(t, (tuple, list)):
        inner = ", ".join(shape_str(x) for x in t)
        tag = "tuple" if isinstance(t, tuple) else "list"
        return f"{tag}[{inner}]"
    return type(t).__name__


def param_count(module: nn.Module) -> str:
    total     = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    if total == 0:
        return ""
    return f"  [{total/1e6:.2f}M params, {trainable/1e6:.2f}M trainable]"


# ══════════════════════════════════════════════════════════════════════
# 1. 正确定位关键模块（直接属性访问，不依赖字符串匹配）
# ══════════════════════════════════════════════════════════════════════

def find_key_modules(model: nn.Module) -> Tuple[Dict, Dict]:
    """
    对 Qwen3VLForConditionalGeneration 使用直接属性访问定位关键模块。

    Qwen3-VL 的结构：
      model (Qwen3VLForConditionalGeneration)
        .model (Qwen3VLModel)
          .visual (Qwen3VLVisionModel)
            .blocks    ← ViT blocks（27层）
            .merger    ← Qwen3VLVisionPatchMerger
            .deepstack_merger_list
          .language_model (Qwen3VLTextModel)
            .embed_tokens
            .layers    ← LLM transformer layers（28层）← SemanticSlot插入点
            .norm
        .lm_head
    """
    result: Dict[str, Optional[nn.Module]] = {
        "visual_model":     None,   # ViT 整体
        "vit_blocks":       None,   # ViT transformer blocks（ModuleList）
        "merger":           None,   # Merger MLP
        "deepstack_mergers":None,   # deepstack_merger_list
        "language_model":   None,   # LLM 整体
        "embed_tokens":     None,   # token embedding
        "llm_layers":       None,   # LLM transformer layers（ModuleList）
        "first_llm_layer":  None,   # LLM 第一层 ← hook 注册位置
        "last_llm_layer":   None,   # LLM 最后一层
        "llm_norm":         None,   # LLM 最终 norm
        "lm_head":          None,   # 语言模型头
    }
    paths: Dict[str, str] = {}

    # ── 获取 base model ──────────────────────────────────────────────
    base = model.model if hasattr(model, "model") else model

    # ── 视觉侧 ───────────────────────────────────────────────────────
    if hasattr(base, "visual"):
        vis = base.visual
        result["visual_model"] = vis
        paths["visual_model"]  = "model.model.visual"

        if hasattr(vis, "blocks"):
            result["vit_blocks"] = vis.blocks
            paths["vit_blocks"]  = "model.model.visual.blocks"

        if hasattr(vis, "merger"):
            result["merger"] = vis.merger
            paths["merger"]  = "model.model.visual.merger"

        if hasattr(vis, "deepstack_merger_list"):
            result["deepstack_mergers"] = vis.deepstack_merger_list
            paths["deepstack_mergers"]  = "model.model.visual.deepstack_merger_list"

    # ── 语言侧 ───────────────────────────────────────────────────────
    if hasattr(base, "language_model"):
        lm = base.language_model
        result["language_model"] = lm
        paths["language_model"]  = "model.model.language_model"

        if hasattr(lm, "embed_tokens"):
            result["embed_tokens"] = lm.embed_tokens
            paths["embed_tokens"]  = "model.model.language_model.embed_tokens"

        if hasattr(lm, "layers") and len(lm.layers) > 0:
            result["llm_layers"]      = lm.layers
            paths["llm_layers"]       = "model.model.language_model.layers"
            result["first_llm_layer"] = lm.layers[0]
            paths["first_llm_layer"]  = "model.model.language_model.layers.0"
            result["last_llm_layer"]  = lm.layers[-1]
            paths["last_llm_layer"]   = f"model.model.language_model.layers.{len(lm.layers)-1}"

        if hasattr(lm, "norm"):
            result["llm_norm"] = lm.norm
            paths["llm_norm"]  = "model.model.language_model.norm"

    # ── lm_head ──────────────────────────────────────────────────────
    if hasattr(model, "lm_head"):
        result["lm_head"] = model.lm_head
        paths["lm_head"]  = "model.lm_head"

    # ── 打印结果 ─────────────────────────────────────────────────────
    logger.info(f"\n{SEP}")
    logger.info("  关键模块定位结果（直接属性访问，修复版）")
    logger.info(SEP)
    for key, path in paths.items():
        mod  = result[key]
        cls  = type(mod).__name__ if mod is not None else "NOT FOUND"
        pc   = param_count(mod) if mod is not None else ""
        logger.info(f"  {key:22s}: [{path}]  {cls}{pc}")

    # 追加 LLM 层数信息
    if result["llm_layers"] is not None:
        n = len(result["llm_layers"])
        logger.info(f"\n  LLM 总层数: {n}")
        logger.info(f"  first_llm_layer class: {type(result['first_llm_layer']).__name__}")

    logger.info(
        f"\n  ★ SemanticSlotToken Hook 注册位置："
        f"\n    模块路径 : {paths.get('first_llm_layer', 'NOT FOUND')}"
        f"\n    模块类型 : {type(result['first_llm_layer']).__name__ if result['first_llm_layer'] else 'NOT FOUND'}"
        f"\n    说明     : 在此注册 pre-forward hook，"
        f"视觉+文本 token 已拼接完毕，插入 K 个 slot token 即可"
        f"\n\n  ★ Merger 位置（slot 计算的输入来源）："
        f"\n    模块路径 : {paths.get('merger', 'NOT FOUND')}"
    )
    logger.info(SEP + "\n")

    return result, paths


# ══════════════════════════════════════════════════════════════════════
# 2. 打印模块树（修复：language_model.layers 单独展开）
# ══════════════════════════════════════════════════════════════════════

def print_module_tree(model: nn.Module, max_depth: int = 4):
    logger.info(f"\n{SEP}")
    logger.info("  模型模块树")
    logger.info(SEP)

    def _print(module, name, depth, fullname):
        if depth > max_depth:
            # 只打印省略提示
            children = list(module.named_children())
            if children and depth == max_depth + 1:
                logger.info(f"{'  '*depth}... ({len(children)} submodules)")
            return
        indent = "  " * depth
        cls    = type(module).__name__
        pc     = param_count(module)
        logger.info(f"{indent}[{fullname}]  {cls}{pc}")
        for child_name, child in module.named_children():
            _print(child, child_name, depth + 1, f"{fullname}.{child_name}")

    _print(model, "model", 0, "model")

    # 单独展开 language_model.layers 前3层（不受 max_depth 限制）
    base = model.model if hasattr(model, "model") else model
    if hasattr(base, "language_model") and hasattr(base.language_model, "layers"):
        layers = base.language_model.layers
        logger.info(f"\n  [language_model.layers 展开，共 {len(layers)} 层，显示前3层]")
        for i, layer in enumerate(layers[:3]):
            logger.info(f"  layers.{i}  {type(layer).__name__}{param_count(layer)}")
            for child_name, child in layer.named_children():
                pc = param_count(child)
                logger.info(f"    .{child_name}  {type(child).__name__}{pc}")

    logger.info(SEP + "\n")


# ══════════════════════════════════════════════════════════════════════
# 3. Shape Tracer（不变）
# ══════════════════════════════════════════════════════════════════════

class TensorShapeTracer:
    def __init__(self):
        self.records: Dict[str, Dict] = {}
        self._hooks: List = []

    def register(self, module: nn.Module, name: str):
        def hook_fn(mod, inputs, outputs):
            self.records[name] = {
                "inputs":  [shape_str(x) for x in inputs]
                           if isinstance(inputs, tuple) else [shape_str(inputs)],
                "outputs": shape_str(outputs),
                "class":   type(mod).__name__,
            }
        self._hooks.append(module.register_forward_hook(hook_fn))

    def remove_all(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def print_records(self):
        logger.info(f"\n{SEP}")
        logger.info("  各关键节点 Tensor Shape（前向传播追踪）")
        logger.info(SEP)
        for name, rec in self.records.items():
            logger.info(f"\n  ▶ {name}  [{rec['class']}]")
            for i, inp in enumerate(rec["inputs"]):
                logger.info(f"      input[{i}] : {inp}")
            logger.info(f"      output   : {rec['outputs']}")
        logger.info(SEP + "\n")


# ══════════════════════════════════════════════════════════════════════
# 4. 视觉 Token 流向追踪（修复 hook 注册在正确模块）
# ══════════════════════════════════════════════════════════════════════

class VisualTokenFlowTracer:
    def __init__(self):
        self.merger_info:       Dict = {}
        self.llm_layer0_info:   Dict = {}
        self.position_ids_info: Dict = {}
        self._hooks: List = []

    def register_merger(self, merger: nn.Module):
        """追踪 Merger 输出的 visual token 数量和维度"""
        def hook(mod, inputs, outputs):
            # outputs 可能是 Tensor 或 tuple
            t = outputs[0] if isinstance(outputs, tuple) else outputs
            if isinstance(t, torch.Tensor):
                self.merger_info = {
                    "shape":            list(t.shape),
                    "num_visual_tokens": t.shape[0],   # (N_vis, D)
                    "dim":              t.shape[-1],
                }
        self._hooks.append(merger.register_forward_hook(hook))

    def register_llm_layer0(self, layer: nn.Module):
        """
        追踪 LLM 第一个 transformer layer 的输入。
        此处 hidden_states 已经是视觉+文本 token 的完整序列。
        这是 SemanticSlotToken 的插入点。
        """
        def hook(mod, inputs, outputs):
            # inputs[0] 是 hidden_states: (B, seq_len, D)
            hs = inputs[0] if isinstance(inputs, tuple) else inputs
            if isinstance(hs, torch.Tensor) and hs.dim() == 3:
                B, seq, D = hs.shape
                self.llm_layer0_info = {
                    "hidden_states_shape": [B, seq, D],
                    "batch_size":          B,
                    "total_seq_len":       seq,
                    "hidden_dim":          D,
                }
                # 记录其他输入（attention_mask, position_ids 等）
                if isinstance(inputs, tuple):
                    for i, inp in enumerate(inputs[1:], 1):
                        if inp is not None:
                            self.llm_layer0_info[f"input_{i}"] = shape_str(inp)
        self._hooks.append(layer.register_forward_hook(hook))

    def register_position_ids(self, model: nn.Module):
        """
        在 Qwen3VLModel.forward 上注册 hook，捕获内部生成的 position_ids。
        用于分析视觉 token 在序列中的索引范围。
        """
        base = model.model if hasattr(model, "model") else model

        def hook(mod, inputs, outputs):
            # 尝试从 inputs kwargs 中找 position_ids
            pass  # 实际由下面的 forward wrapper 实现

        # 替代方案：monkey-patch get_rope_index 捕获输出
        visual = getattr(base, "visual", None)
        if visual is not None:
            orig_forward = visual.forward

            tracer = self

            def patched_forward(*args, **kwargs):
                result = orig_forward(*args, **kwargs)
                # result 通常是 (visual_tokens,) 或 visual_tokens
                t = result[0] if isinstance(result, tuple) else result
                if isinstance(t, torch.Tensor):
                    tracer.merger_info["visual_output_shape"] = list(t.shape)
                return result

            visual.forward = patched_forward
            self._patched_visual = (visual, orig_forward)

    def restore_patches(self):
        if hasattr(self, "_patched_visual"):
            vis, orig = self._patched_visual
            vis.forward = orig

    def remove_all(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.restore_patches()

    def print_analysis(
        self,
        num_text_tokens: int,
        use_position_info: bool = False,
    ):
        logger.info(f"\n{SEP}")
        logger.info("  视觉 Token 流向分析（修复版）")
        logger.info(SEP)

        logger.info("\n  [Merger 输出]")
        if self.merger_info:
            for k, v in self.merger_info.items():
                logger.info(f"    {k}: {v}")
        else:
            logger.info("    ⚠ 未捕获到数据（前向传播可能提前失败）")

        logger.info("\n  [LLM 第一层输入（SemanticSlotToken 插入点）]")
        if self.llm_layer0_info:
            for k, v in self.llm_layer0_info.items():
                logger.info(f"    {k}: {v}")
        else:
            logger.info("    ⚠ 未捕获到数据（前向传播可能提前失败）")

        # 序列结构推算
        total_seq  = self.llm_layer0_info.get("total_seq_len", 0)
        num_visual = self.merger_info.get("num_visual_tokens", 0)
        if total_seq > 0:
            logger.info("\n  [序列结构推算]")
            logger.info(f"    total_seq_len      = {total_seq}")
            logger.info(f"    num_text_tokens    ≈ {num_text_tokens}  (system prompt + user prompt)")
            logger.info(f"    num_visual_tokens  = {num_visual}")
            logger.info(f"    其他特殊tokens     ≈ {total_seq - num_text_tokens - num_visual}")

        # SemanticSlotToken 插入后的序列变化
        K_examples = [4, 8, 12]
        logger.info("\n  [SemanticSlotToken 插入后序列长度变化]")
        for K in K_examples:
            logger.info(f"    K={K:2d} slots: {total_seq} → {total_seq + K}")

        # 位置编码分析
        if use_position_info:
            logger.info("\n  [位置编码分析（--use_position_info 已开启）]")
            logger.info(
                "    Qwen3-VL 使用 3D RoPE（t/h/w 三维），position_ids 在模型内部计算。"
                "\n    SemanticSlotToken 的 slot token 位置编码策略："
                "\n      方案A：追加在视觉token末尾，position_ids 设为视觉区间末尾+1 起的连续值"
                "\n      方案B：position_ids 设为固定的可学习偏移量（与空间坐标解耦）"
                "\n      推荐：方案A，与现有 rope_index 体系保持一致，实现简单"
            )
        else:
            logger.info(
                "\n  [位置编码] 使用 --use_position_info 查看详细位置编码分析"
            )

        logger.info(f"\n{SEP}\n")


# ══════════════════════════════════════════════════════════════════════
# 5. 分阶段前向传播（修复 "not enough values to unpack" 错误）
# ══════════════════════════════════════════════════════════════════════

def run_forward_staged(
    model: nn.Module,
    inputs: Dict,
    key_modules: Dict,
    tracer: TensorShapeTracer,
    vf_tracer: VisualTokenFlowTracer,
) -> bool:
    """
    分三个阶段执行前向传播，每阶段独立 try-catch，
    最大程度获取调试信息。
    """
    success = False

    # ── 阶段1：仅运行视觉编码器 ────────────────────────────────────
    logger.info("阶段1：运行视觉编码器（ViT + Merger）...")
    try:
        base    = model.model if hasattr(model, "model") else model
        visual  = base.visual

        pv  = inputs.get("pixel_values")
        thw = inputs.get("image_grid_thw")

        if pv is not None and thw is not None:
            with torch.no_grad():
                vis_out = visual(pv, grid_thw=thw)
            t = vis_out[0] if isinstance(vis_out, tuple) else vis_out
            if isinstance(t, torch.Tensor):
                logger.info(f"  ✅ 视觉编码器输出 shape: {list(t.shape)}")
                vf_tracer.merger_info["num_visual_tokens"] = t.shape[0]
                vf_tracer.merger_info["dim"]               = t.shape[-1]
                vf_tracer.merger_info["shape"]             = list(t.shape)
        else:
            logger.info("  ⚠ 无 pixel_values，跳过视觉编码器")
    except Exception as e:
        logger.warning(f"  ⚠ 阶段1 出错: {e}")

    # ── 阶段2：运行完整前向（修复输入构造）────────────────────────
    logger.info("阶段2：运行完整模型前向...")
    try:
        # 只传递模型接受的标准参数，避免多余参数导致解包错误
        forward_inputs = {
            k: v for k, v in inputs.items()
            if k in ("input_ids", "attention_mask",
                     "pixel_values", "image_grid_thw",
                     "pixel_values_videos", "video_grid_thw")
        }
        with torch.no_grad():
            outputs = model(**forward_inputs, return_dict=True)
        logger.info(f"  ✅ 完整前向成功，logits shape: {list(outputs.logits.shape)}")
        success = True
    except TypeError as e:
        # 可能是参数不匹配，尝试最小化输入
        logger.warning(f"  ⚠ 阶段2 TypeError: {e}")
        logger.info("  尝试最小化输入（仅 input_ids + attention_mask）...")
        try:
            min_inputs = {
                "input_ids":      inputs["input_ids"],
                "attention_mask": inputs.get("attention_mask"),
            }
            with torch.no_grad():
                outputs = model(**min_inputs, return_dict=True)
            logger.info(f"  ✅ 最小化输入前向成功，logits shape: {list(outputs.logits.shape)}")
            success = True
        except Exception as e2:
            logger.warning(f"  ⚠ 最小化输入也失败: {e2}")
    except Exception as e:
        logger.warning(f"  ⚠ 阶段2 出错: {e}")
        logger.info(f"  错误类型: {type(e).__name__}")

    # ── 阶段3：追踪 embed_tokens 输出（获取 LLM 输入维度）────────
    logger.info("阶段3：追踪 embed_tokens 维度（推断 LLM hidden_dim）...")
    try:
        embed = key_modules.get("embed_tokens")
        if embed is not None:
            # 只用文本 input_ids 测试 embedding
            text_ids = inputs["input_ids"]
            with torch.no_grad():
                emb_out = embed(text_ids)
            B, seq, D = emb_out.shape
            logger.info(f"  ✅ embed_tokens 输出 shape: [{B}, {seq}, {D}]")
            logger.info(f"     → LLM hidden_dim = {D}")
            vf_tracer.llm_layer0_info["hidden_dim"]   = D
            vf_tracer.llm_layer0_info["total_seq_len"] = \
                vf_tracer.merger_info.get("num_visual_tokens", 0) + seq
            vf_tracer.llm_layer0_info["batch_size"]   = B
        else:
            logger.warning("  ⚠ 未找到 embed_tokens")
    except Exception as e:
        logger.warning(f"  ⚠ 阶段3 出错: {e}")

    return success


# ══════════════════════════════════════════════════════════════════════
# 6. position_ids 详细分析（--use_position_info 开启时）
# ══════════════════════════════════════════════════════════════════════

def analyze_position_ids_detailed(model: nn.Module, inputs: Dict):
    """
    当 --use_position_info 开启时，详细分析 Qwen3-VL 的 3D RoPE 结构。
    通过 monkey-patch 模型内部的 get_rope_index 函数捕获 position_ids。
    """
    logger.info(f"\n{SEP}")
    logger.info("  Position IDs 详细分析（3D RoPE）")
    logger.info(SEP)

    captured_pos_ids = {}

    # 尝试 patch Qwen3VLModel 的 get_rope_index
    base = model.model if hasattr(model, "model") else model
    orig_rope_fn = None

    # 查找 get_rope_index 函数
    if hasattr(base, "get_rope_index"):
        orig_rope_fn = base.get_rope_index

        def patched_rope(*args, **kwargs):
            result = orig_rope_fn(*args, **kwargs)
            if isinstance(result, (tuple, list)) and len(result) >= 1:
                pos = result[0]
                if isinstance(pos, torch.Tensor):
                    captured_pos_ids["position_ids"] = pos
                    captured_pos_ids["shape"]        = list(pos.shape)
                    captured_pos_ids["dtype"]        = str(pos.dtype)
                    if pos.dim() == 3:
                        # 3D RoPE: (3, B, seq_len) 或 (B, 3, seq_len)
                        captured_pos_ids["dim_t_range"] = \
                            [pos[0].min().item(), pos[0].max().item()]
                        captured_pos_ids["dim_h_range"] = \
                            [pos[1].min().item(), pos[1].max().item()]
                        captured_pos_ids["dim_w_range"] = \
                            [pos[2].min().item(), pos[2].max().item()]
            return result

        base.get_rope_index = patched_rope

    # 运行一次前向
    try:
        forward_inputs = {
            k: v for k, v in inputs.items()
            if k in ("input_ids", "attention_mask",
                     "pixel_values", "image_grid_thw")
        }
        with torch.no_grad():
            model(**forward_inputs, return_dict=True)
    except Exception as e:
        logger.warning(f"  前向传播出错（position_ids 分析）: {e}")

    # 恢复
    if orig_rope_fn is not None:
        base.get_rope_index = orig_rope_fn

    # 打印
    if captured_pos_ids:
        logger.info(f"  position_ids shape: {captured_pos_ids.get('shape')}")
        logger.info(f"  dtype: {captured_pos_ids.get('dtype')}")
        if "dim_t_range" in captured_pos_ids:
            logger.info(f"  dim_t (时序) 范围: {captured_pos_ids['dim_t_range']}")
            logger.info(f"  dim_h (高度) 范围: {captured_pos_ids['dim_h_range']}")
            logger.info(f"  dim_w (宽度) 范围: {captured_pos_ids['dim_w_range']}")

            # 推断视觉 token 位置
            pos = captured_pos_ids.get("position_ids")
            if pos is not None and pos.dim() == 3:
                # 视觉 token 特征：h 和 w 维度非零
                h_vals = pos[1]  # (B, seq) 或 (seq,)
                w_vals = pos[2]
                if h_vals.dim() == 2:
                    h_vals = h_vals[0]
                    w_vals = w_vals[0]
                vis_mask    = (h_vals != 0) | (w_vals != 0)
                vis_indices = vis_mask.nonzero(as_tuple=True)[0]
                if len(vis_indices) > 0:
                    logger.info(
                        f"\n  视觉 token 索引范围: "
                        f"[{vis_indices.min().item()}, {vis_indices.max().item()}]"
                        f"  共 {len(vis_indices)} 个"
                    )
                    logger.info(
                        f"  文本 token 索引: 其余 "
                        f"{pos.shape[-1] - len(vis_indices)} 个位置"
                    )
                    logger.info(
                        f"\n  [SemanticSlotToken 位置编码建议]"
                        f"\n    slot token 追加在序列末尾，position_ids 设置："
                        f"\n      t 维度: {captured_pos_ids['dim_t_range'][1] + 1} 起连续"
                        f"\n      h 维度: 0（slot 无空间高度含义）"
                        f"\n      w 维度: 0~K-1（K 个 slot 的序号）"
                    )
    else:
        logger.info("  ⚠ 未能捕获 position_ids，可能 get_rope_index 路径不同")
        logger.info("  attention_mask shape: " +
                    str(list(inputs.get("attention_mask", torch.tensor([])).shape)))

    logger.info(SEP + "\n")


# ══════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL 模型结构调试工具（修复版）")
    parser.add_argument("--model_path",        required=True)
    parser.add_argument("--image_path",        default=None)
    parser.add_argument("--use_position_info", action="store_true",
                        help="开启位置编码详细分析（为 SemanticSlotToken 位置分支准备）")
    parser.add_argument("--verbose_tensor",    action="store_true",
                        help="打印所有关键节点的 tensor shape")
    parser.add_argument("--max_depth",         type=int, default=4)
    parser.add_argument("--output_json",       default=None)
    args = parser.parse_args()

    # ── 加载模型（CPU 调试） ──────────────────────────────────────────
    logger.info(f"加载模型: {args.model_path}  （CPU 模式）")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    model.eval()
    logger.info("模型加载完成\n")

    # ── Step1：打印模块树 ─────────────────────────────────────────────
    print_module_tree(model, max_depth=args.max_depth)

    # ── Step2：定位关键模块（修复版） ──────────────────────────────────
    key_modules, key_paths = find_key_modules(model)

    # ── Step3：构造测试输入 ───────────────────────────────────────────
    if args.image_path and Path(args.image_path).exists():
        logger.info(f"使用图片: {args.image_path}")
        image = Image.open(args.image_path).convert("RGB")
    else:
        import numpy as np
        logger.info("未指定图片，使用随机 512×512 图片")
        image = Image.fromarray(
            (np.random.rand(512, 512, 3) * 255).astype("uint8")
        )

    prompt = "Please describe this image in detail."
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": prompt},
        ],
    }]
    text_fmt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=text_fmt, images=[image], return_tensors="pt")

    num_text_tokens = len(processor.tokenizer.encode(prompt))
    logger.info(f"input_ids shape:    {list(inputs['input_ids'].shape)}")
    logger.info(f"attention_mask shape: {list(inputs['attention_mask'].shape)}")
    if "pixel_values" in inputs:
        logger.info(f"pixel_values shape: {list(inputs['pixel_values'].shape)}")
    if "image_grid_thw" in inputs:
        logger.info(f"image_grid_thw:     {inputs['image_grid_thw']}")
    logger.info(f"prompt token 数:    {num_text_tokens}\n")

    # ── Step4：注册 tracer hooks ──────────────────────────────────────
    tracer    = TensorShapeTracer()
    vf_tracer = VisualTokenFlowTracer()

    if key_modules["merger"] is not None:
        tracer.register(key_modules["merger"], "merger")
        vf_tracer.register_merger(key_modules["merger"])

    if key_modules["first_llm_layer"] is not None:
        tracer.register(key_modules["first_llm_layer"], "first_llm_layer")
        vf_tracer.register_llm_layer0(key_modules["first_llm_layer"])
    else:
        logger.warning("⚠ first_llm_layer 未找到，无法注册 LLM 层 hook")

    if args.verbose_tensor:
        for k in ("visual_model", "embed_tokens", "llm_norm", "lm_head"):
            if key_modules.get(k) is not None:
                tracer.register(key_modules[k], k)

    # ── Step5：分阶段前向传播 ─────────────────────────────────────────
    logger.info("开始分阶段前向传播...")
    success = run_forward_staged(
        model, inputs, key_modules, tracer, vf_tracer
    )

    # ── Step6：打印 tracer 结果 ───────────────────────────────────────
    if args.verbose_tensor:
        tracer.print_records()

    vf_tracer.print_analysis(
        num_text_tokens=num_text_tokens,
        use_position_info=args.use_position_info,
    )

    # ── Step7：位置编码详细分析（可选）───────────────────────────────
    if args.use_position_info:
        analyze_position_ids_detailed(model, inputs)

    # ── Step8：清理 hooks ─────────────────────────────────────────────
    tracer.remove_all()
    vf_tracer.remove_all()

    # ── Step9：汇总 ──────────────────────────────────────────────────
    summary = {
        "model_path":            args.model_path,
        "use_position_info":     args.use_position_info,
        "forward_pass_success":  success,
        "key_module_paths":      key_paths,
        "input_shapes": {
            "input_ids":      list(inputs["input_ids"].shape),
            "pixel_values":   list(inputs["pixel_values"].shape)
                              if "pixel_values" in inputs else None,
            "image_grid_thw": inputs["image_grid_thw"].tolist()
                              if "image_grid_thw" in inputs else None,
        },
        "visual_token_info":   vf_tracer.merger_info,
        "llm_input_info":      vf_tracer.llm_layer0_info,
        "hook_insert_point": {
            "module_path":  key_paths.get("first_llm_layer", "NOT FOUND"),
            "module_class": type(key_modules["first_llm_layer"]).__name__
                            if key_modules["first_llm_layer"] else "NOT FOUND",
            "note": (
                "在此注册 pre-forward hook。"
                "hidden_states 是视觉+文本 token 已拼接的完整序列。"
                "SemanticSlotToken 在此追加 K 个 slot token。"
            ),
        },
        "semantic_slot_implementation_notes": {
            "slot_input":   "Merger 输出的 visual tokens，形状 (N_vis, D)",
            "slot_output":  "K 个 slot tokens，形状 (K, D)，追加到序列末尾",
            "seq_change":   f"seq_len: {vf_tracer.llm_layer0_info.get('total_seq_len', '?')} "
                            f"→ {vf_tracer.llm_layer0_info.get('total_seq_len', '?')}+K",
            "position_info_branch": (
                "已开启 --use_position_info：slot token 使用空间重心坐标"
                if args.use_position_info
                else "未开启 --use_position_info：slot token 使用追加索引位置编码"
            ),
        },
    }

    logger.info(f"\n{SEP}")
    logger.info("  完整分析摘要")
    logger.info(SEP)
    logger.info(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"\n结果已保存: {args.output_json}")

    logger.info(
        "\n[完成]"
        "\n  下一步：确认 hook_insert_point 后，开始实现 semantic_slot.py"
        "\n  关键参数："
        f"\n    LLM hidden_dim  = {vf_tracer.llm_layer0_info.get('hidden_dim', '待确认')}"
        f"\n    num_visual_tokens = {vf_tracer.merger_info.get('num_visual_tokens', '待确认')}"
        f"\n    total_seq_len   = {vf_tracer.llm_layer0_info.get('total_seq_len', '待确认')}"
    )


if __name__ == "__main__":
    main()