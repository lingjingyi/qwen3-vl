"""
诊断 slot 梯度为什么断流 - 不跑训练,直接测梯度
"""
import sys
sys.path.insert(0, '/opt/data/private/qwen3-vl-master/qwen3-vl')

import torch
from qwenvl.model.rs_caption_model import RSCaptionModel, VISION_END_ID
from qwenvl.model.semantic_slot import SemanticSlotModule

MP = "/opt/data/private/qwen3-vl-master/qwen3-vl/pretrained/Qwen3-VL-8B-Instruct"

print("加载模型到 cuda:0 (禁用 offload)...")
# ★ 不用 device_map=auto, 避免 CPU offload 导致 backward 失败
model = RSCaptionModel.from_pretrained(
    MP, torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to("cuda")
model.config.use_cache = False

# 打印关键 visual 配置, 用于构造正确的 pixel_values shape
vc = model.config.vision_config
print(f"\nvision_config:")
print(f"  patch_size            = {getattr(vc, 'patch_size', 'N/A')}")
print(f"  spatial_merge_size    = {getattr(vc, 'spatial_merge_size', 'N/A')}")
print(f"  in_channels           = {getattr(vc, 'in_channels', 'N/A')}")
print(f"  temporal_patch_size   = {getattr(vc, 'temporal_patch_size', 'N/A')}")
print(f"  hidden_size           = {getattr(vc, 'hidden_size', 'N/A')}")

# 从 config 算出每个 patch 的数据长度 (给 patch_embed 的输入)
in_ch   = getattr(vc, "in_channels", 3)
ps      = getattr(vc, "patch_size", 16)
tps     = getattr(vc, "temporal_patch_size", 2)
patch_len = in_ch * tps * ps * ps   # 例如 3 * 2 * 16 * 16 = 1536
print(f"\n每个 patch 向量长度 = {in_ch} * {tps} * {ps} * {ps} = {patch_len}")

# ── 挂载 slot ──
slot = SemanticSlotModule(num_slots=9, dim=4096, use_count_head=False)
slot = slot.to("cuda", torch.bfloat16)
for p in slot.parameters():
    p.requires_grad = True

model.attach_modules(slot_module=slot, style_module=None)

# ── 冻住其他, 只让 slot 有梯度 ──
for p in model.parameters():
    p.requires_grad = False
for p in slot.parameters():
    p.requires_grad = True

model.train()
model.zero_grad()
slot.zero_grad()

# ── 构造假 batch ──
# grid_thw = [1, 8, 10] 意味着 t=1, h=8 pre-merge patches, w=10 pre-merge patches
# total pre-merge patches = 1 * 8 * 10 = 80
# post-merge tokens = 80 / (spatial_merge_size^2) = 80 / 4 = 20 (假设 merge_size=2)
# 所以 input_ids 里要有 20 个 image_token
B, seq_len, vis_end_pos = 1, 80, 30

ms = getattr(vc, "spatial_merge_size", 2)
n_patches_pre  = 1 * 8 * 10              # 80
n_patches_post = n_patches_pre // (ms * ms)   # 20
print(f"\ngrid_thw=[1,8,10], pre-merge={n_patches_pre}, post-merge={n_patches_post}")

input_ids = torch.randint(100, 30000, (B, seq_len), device="cuda")
input_ids[:, vis_end_pos] = VISION_END_ID

image_token_id = model.config.image_token_id
# 占位 20 个 image_token
input_ids[:, 5 : 5 + n_patches_post] = image_token_id

attention_mask = torch.ones_like(input_ids)
labels = input_ids.clone()
labels[:, :vis_end_pos+1] = -100

base_pos = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(B, -1)
position_ids = torch.stack([base_pos, base_pos*0, base_pos*0], dim=0).clone()

pixel_values   = torch.randn(n_patches_pre, patch_len, device="cuda", dtype=torch.bfloat16)
image_grid_thw = torch.tensor([[1, 8, 10]], device="cuda")

print(f"\npixel_values shape = {list(pixel_values.shape)}")
print(f"image_grid_thw     = {image_grid_thw.tolist()}")

print("\nforward...")
out = model(
    input_ids      = input_ids,
    attention_mask = attention_mask,
    position_ids   = position_ids,
    labels         = labels,
    pixel_values   = pixel_values,
    image_grid_thw = image_grid_thw,
)
print(f"loss = {out.loss.item():.4f}")

print("\nbackward...")
out.loss.backward()

# ── 检查每个 slot 参数的梯度 ──
print("\n" + "=" * 70)
print("slot 参数梯度检查 (关键: 看 slot_init 和 cross_attn)")
print("=" * 70)

categories = {
    "slot_init":              [],
    "slot_attn (self)":       [],
    "slot_refine (cross):":   [],
    "slot_refine (ffn):":     [],
    "spatial_bias":           [],
}

for n, p in slot.named_parameters():
    if p.grad is None:
        status, gm = "❌ None", None
    else:
        gm = p.grad.abs().mean().item()
        status = f"✅ {gm:.3e}" if gm > 1e-9 else f"❌ {gm:.3e}"

    if "slot_init" in n:
        categories["slot_init"].append((n, status, gm))
    elif "slot_refine.cross_attn" in n or "slot_refine.norm_vis" in n or "slot_refine.norm_slots" in n:
        categories["slot_refine (cross):"].append((n, status, gm))
    elif "slot_refine.ffn" in n or "slot_refine.norm_ffn" in n or "slot_refine.refine_scale" in n:
        categories["slot_refine (ffn):"].append((n, status, gm))
    elif "slot_attn" in n:
        categories["slot_attn (self)"].append((n, status, gm))
    elif "spatial_bias" in n:
        categories["spatial_bias"].append((n, status, gm))

for cat, items in categories.items():
    print(f"\n-- {cat} --")
    for n, s, _ in items:
        print(f"  {s:20s}  {n}")

# ── 结论 ──
print("\n" + "=" * 70)
print("结论")
print("=" * 70)

si = slot.slot_attn.slot_init
rs = slot.slot_refine.refine_scale
cross_q = slot.slot_refine.cross_attn.in_proj_weight
ffn_0   = slot.slot_refine.ffn[0].weight

si_g    = si.grad.abs().mean().item() if si.grad is not None else 0
rs_g    = rs.grad.abs().mean().item() if rs.grad is not None else 0
cross_g = cross_q.grad.abs().mean().item() if cross_q.grad is not None else 0
ffn_g   = ffn_0.grad.abs().mean().item() if ffn_0.grad is not None else 0

print(f"  slot_init   梯度: {si_g:.3e}   {'✅' if si_g > 1e-9 else '❌'}")
print(f"  refine_scale梯度: {rs_g:.3e}   {'✅' if rs_g > 1e-9 else '❌'}")
print(f"  cross_attn  梯度: {cross_g:.3e}   {'✅' if cross_g > 1e-9 else '❌'}")
print(f"  ffn[0]      梯度: {ffn_g:.3e}   {'✅' if ffn_g > 1e-9 else '❌'}")

if cross_g < 1e-9 and ffn_g > 1e-9:
    print("\n  🎯 诊断: refine_scale=0 截断了 cross_attn 分支梯度 (情况 A)")
    print("     修复: 改 refine_scale 初值为 0.05")
elif si_g < 1e-9 and cross_g < 1e-9 and ffn_g < 1e-9:
    print("\n  🎯 诊断: slot 完全无梯度 — refine_visual_tokens 没被调用 (情况 B)")
    print("     需要检查 _process_visual 是否真的调了 slot")
elif si_g > 1e-9 and cross_g > 1e-9:
    print("\n  🎯 诊断: 梯度正常, slot 应该能学. 问题在别处 (DeepSpeed / lr / scheduler)")