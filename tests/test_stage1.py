"""
tests/test_stage1.py  (方案 D 验证脚本)

验证方案 D 的核心改动:
  Test 1 - slot_init + style_embeds 梯度同时流通
           (关键: slot 现在通过 LLM attention 直接反传, 梯度应该显著 > 旧版)
  Test 2 - position_ids 长度 = orig + K_slot + K_style, 时间轴单调递增
  Test 3 - 100 次 forward 无 patch 污染
  Test 4 - generate 兼容性

运行:
  python tests/test_stage1.py \
      --model_path /opt/data/private/qwen3-vl-master/qwen3-vl/pretrained/Qwen3-VL-8B-Instruct
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from qwenvl.model.rs_caption_model import RSCaptionModel, VISION_END_ID
from qwenvl.model.style_prefix import StylePrefixModule
from qwenvl.model.semantic_slot import SemanticSlotModule

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_test_model(model_path: str, attach_style=True, attach_slot=True):
    print(f"[Setup] 加载 RSCaptionModel from {model_path} ...")
    print(f"[Setup] 目标设备: {DEVICE}")

    load_kwargs = dict(torch_dtype=torch.bfloat16)
    if DEVICE == "cuda":
        load_kwargs["device_map"] = "auto"
        load_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        load_kwargs["device_map"] = "cpu"

    model = RSCaptionModel.from_pretrained(model_path, **load_kwargs)
    model.config.use_cache = False

    dim = model.config.text_config.hidden_size
    print(f"[Setup] 模型维度 dim={dim}")

    slot = None
    if attach_slot:
        slot = SemanticSlotModule(
            num_slots=9, dim=dim, use_count_head=False, use_position_info=True
        )
        slot = slot.to(device=DEVICE, dtype=torch.bfloat16)
        for p in slot.parameters():
            p.requires_grad = True

    style = None
    if attach_style:
        style = StylePrefixModule(num_style_tokens=8, dim=dim, use_slot_attention=False)
        style = style.to(device=DEVICE, dtype=torch.bfloat16)
        for p in style.parameters():
            p.requires_grad = True

    model.attach_modules(slot_module=slot, style_module=style)
    print("[Setup] 模块挂载完成\n")
    return model


def build_dummy_batch_with_visual(
    model,
    batch_size: int = 2,
    seq_len:    int = 80,
    vision_end_pos: int = 30,
):
    """
    构造假 batch 含真实的 pixel_values, 让 slot 能真的走 compute_slots 路径.
    """
    # ── text side ──
    image_token_id = model.config.image_token_id
    input_ids = torch.randint(100, 30000, (batch_size, seq_len), device=DEVICE)
    input_ids[:, vision_end_pos] = VISION_END_ID
    # 放 20 个 image_token (用于 masked_scatter, 匹配 post-merger tokens 数)
    input_ids[:, 5:25] = image_token_id

    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :vision_end_pos + 1] = -100

    base_pos = torch.arange(seq_len, device=DEVICE).unsqueeze(0).expand(batch_size, -1)
    position_ids = torch.stack([base_pos, base_pos * 0, base_pos * 0], dim=0).clone()

    # ── pixel side ──
    # grid_thw = [1, 8, 10] → pre-merge 80 patches, post-merge 20 tokens (merge_size=2)
    vc = model.config.vision_config
    ps  = getattr(vc, "patch_size", 16)
    tps = getattr(vc, "temporal_patch_size", 2)
    in_ch = getattr(vc, "in_channels", 3)
    patch_len = in_ch * tps * ps * ps

    n_patches_pre = 1 * 8 * 10  # 80
    # batch_size 个样本 → batch_size × 80 个 patches
    pixel_values = torch.randn(
        batch_size * n_patches_pre, patch_len,
        device=DEVICE, dtype=torch.bfloat16,
    )
    image_grid_thw = torch.tensor(
        [[1, 8, 10]] * batch_size, device=DEVICE
    )

    return dict(
        input_ids      = input_ids,
        attention_mask = attention_mask,
        labels         = labels,
        position_ids   = position_ids,
        pixel_values   = pixel_values,
        image_grid_thw = image_grid_thw,
    )


def build_dummy_batch_text_only(batch_size=2, seq_len=80, vision_end_pos=30):
    """用于 Test 2/3: 只需要 text side, 不需要真实 visual"""
    input_ids = torch.randint(100, 30000, (batch_size, seq_len), device=DEVICE)
    input_ids[:, vision_end_pos] = VISION_END_ID
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :vision_end_pos + 1] = -100
    base_pos = torch.arange(seq_len, device=DEVICE).unsqueeze(0).expand(batch_size, -1)
    position_ids = torch.stack([base_pos, base_pos * 0, base_pos * 0], dim=0).clone()
    return dict(
        input_ids      = input_ids,
        attention_mask = attention_mask,
        labels         = labels,
        position_ids   = position_ids,
    )


def test_gradient_flow(model):
    print("=" * 60)
    print("Test 1: slot_init + style_embeds 梯度流通 (方案 D 关键)")
    print("=" * 60)

    model.train()
    model.zero_grad()

    batch = build_dummy_batch_with_visual(model)

    out = model(**batch)

    assert out.loss is not None, "loss 为 None!"
    assert not out.loss.isnan(), f"loss 是 NaN!"
    out.loss.backward()

    # ── 检查 style_embeds 梯度 ──
    style_grad = model.style_prefix.style_embeds.grad
    assert style_grad is not None, \
        "❌ style_embeds.grad is None"
    style_gm = style_grad.abs().mean().item()
    assert style_gm > 1e-9, \
        f"❌ style_embeds.grad 过小: {style_gm:.2e}"
    print(f"  ✅ style_embeds.grad.abs().mean() = {style_gm:.4e}")

    # ── 检查 slot_init 梯度 (方案 D 核心测试) ──
    slot_grad = model.semantic_slot.slot_attn.slot_init.grad
    assert slot_grad is not None, \
        "❌ slot_init.grad is None — 方案 D 失败, slot 梯度未流通!"
    slot_gm = slot_grad.abs().mean().item()
    assert slot_gm > 1e-9, \
        f"❌ slot_init.grad 过小: {slot_gm:.2e}"
    print(f"  ✅ slot_init.grad.abs().mean() = {slot_gm:.4e}")
    print(f"     (旧版 B2 这里大约 1.7e-5, 方案 D 期望 >= 1e-4, 提升 10×+)")

    # 额外诊断: slot 内部其他参数
    for name_sub in ["q_proj.weight", "k_proj.weight", "v_proj.weight"]:
        p = dict(model.semantic_slot.slot_attn.named_parameters()).get(name_sub, None)
        if p is not None and p.grad is not None:
            print(f"     slot_attn.{name_sub}.grad: {p.grad.abs().mean():.4e}")

    model.zero_grad()
    print("  ✅ Test 1 PASS\n")


def test_position_monotonic(model):
    print("=" * 60)
    print("Test 2: position_ids 长度 + 单调递增")
    print("=" * 60)

    model.eval()
    batch = build_dummy_batch_with_visual(model)

    captured = {}
    orig_lm = model.model.language_model.forward

    def capture_lm(*args, **kwargs):
        captured["position_ids"] = kwargs.get("position_ids")
        captured["inputs_embeds_shape"] = (
            kwargs["inputs_embeds"].shape if kwargs.get("inputs_embeds") is not None else None
        )
        return orig_lm(*args, **kwargs)
    model.model.language_model.forward = capture_lm

    try:
        with torch.no_grad():
            model(**batch)
    finally:
        model.model.language_model.forward = orig_lm

    pos = captured.get("position_ids")
    assert pos is not None, "position_ids 未被传入 language_model!"

    K_slot  = model.semantic_slot.num_slots
    K_style = model.style_prefix.num_style_tokens
    K_total = K_slot + K_style
    orig_seq = batch["input_ids"].shape[1]
    expected_len = orig_seq + K_total

    print(f"  K_slot = {K_slot}, K_style = {K_style}, K_total = {K_total}")
    print(f"  position_ids shape: {pos.shape}")
    print(f"  inputs_embeds shape: {captured['inputs_embeds_shape']}")

    assert pos.shape[2] == expected_len, \
        f"❌ position_ids 长度 {pos.shape[2]} ≠ orig_seq+K_total={expected_len}"
    print(f"  ✅ 序列长度正确: orig({orig_seq}) + K_slot({K_slot}) + K_style({K_style}) = {expected_len}")

    # 时间轴单调递增
    for b in range(pos.shape[1]):
        diff = pos[0, b, 1:] - pos[0, b, :-1]
        bad = (diff < 0).sum().item()
        assert bad == 0, \
            f"❌ batch {b} 时间轴有 {bad} 处下降"
    print(f"  ✅ 时间轴严格单调递增 (batch={pos.shape[1]})")
    print("  ✅ Test 2 PASS\n")


def test_no_patch_pollution(model):
    print("=" * 60)
    print("Test 3: 100 次 forward 无 patch 污染")
    print("=" * 60)

    original_lm_forward = model.model.language_model.forward

    model.eval()
    batch = build_dummy_batch_with_visual(model)

    for i in range(100):
        with torch.no_grad():
            model(**batch)

    assert model.model.language_model.forward is original_lm_forward, \
        "❌ language_model.forward 被污染!"

    print("  ✅ 100 次 forward 后, language_model.forward 未被污染")
    print("  ✅ Test 3 PASS\n")


def test_generate_compatibility(model):
    print("=" * 60)
    print("Test 4: generate 兼容性")
    print("=" * 60)

    model.eval()
    batch = build_dummy_batch_with_visual(model, batch_size=1, seq_len=50, vision_end_pos=20)

    with torch.no_grad():
        out = model.generate(
            input_ids      = batch["input_ids"],
            attention_mask = batch["attention_mask"],
            pixel_values   = batch["pixel_values"],
            image_grid_thw = batch["image_grid_thw"],
            max_new_tokens = 10,
            do_sample      = False,
            use_cache      = True,
        )

    assert out.shape[0] == 1, f"batch 维度错误: {out.shape[0]}"
    assert out.shape[1] > batch["input_ids"].shape[1], "generate 未生成任何新 token"
    print(f"  ✅ generate 成功, output shape: {out.shape}")
    print(f"  ✅ 生成了 {out.shape[1] - batch['input_ids'].shape[1]} 个新 token")
    print("  ✅ Test 4 PASS\n")


def main():
    parser = argparse.ArgumentParser(description="方案 D 验证脚本")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--skip_generate", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    global DEVICE
    if args.device:
        DEVICE = args.device
    else:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "█" * 60)
    print("方案 D 验证: Slot + Style 作为 LLM Context Tokens")
    print("█" * 60 + "\n")

    if DEVICE == "cpu":
        print("⚠️  CPU 运行很慢, 建议 GPU\n")

    model = build_test_model(args.model_path)

    passed = 0
    failed = 0

    tests = [
        ("梯度流通",       lambda: test_gradient_flow(model)),
        ("position单调",   lambda: test_position_monotonic(model)),
        ("幂等性",         lambda: test_no_patch_pollution(model)),
    ]
    if not args.skip_generate:
        tests.append(("generate", lambda: test_generate_compatibility(model)))

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ FAIL [{name}]: {e}\n")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  ❌ ERROR [{name}]: {e}")
            traceback.print_exc()
            print()
            failed += 1

    print("█" * 60)
    if failed == 0:
        print(f"✅ 全部 {passed} 个测试通过！可以训练方案 D")
    else:
        print(f"❌ {failed} 个测试失败, {passed} 个通过")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    main()