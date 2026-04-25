"""快速验证 style_embeds init 的多样性"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer
from qwenvl.model.rs_caption_model import RSCaptionModel
from qwenvl.model.style_prefix import StylePrefixModule

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True)
args = parser.parse_args()

SENTENCES = [
    "This is a high-resolution satellite image showing a large expanse of grassland.",
    "This is a high-resolution aerial image. In the bottom left corner of the image, there is a parking lot with seven long yellow buses parked neatly in a row.",
    "This is a panchromatic satellite image showing a large dense residential area with many square houses arranged in a neat pattern.",
    "This is a high-resolution remote sensing image. In the center of the image, there is a large lake, and the water is green.",
    "This is a high-resolution aerial image showing a residential area. Two roads running east to west cross the residential area.",
    "This is an aerial image. In the bottom-right corner of the image, there is a body of water with two slender ports extending out.",
]

tok   = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
model = RSCaptionModel.from_pretrained(
    args.model_path, torch_dtype=torch.bfloat16, device_map="auto"
)
embed_fn = model.model.language_model.embed_tokens

style = StylePrefixModule(num_style_tokens=6, dim=4096, use_slot_attention=True)
style = style.to(device="cuda", dtype=torch.bfloat16)

style.initialize_from_sentences(SENTENCES, tok, embed_fn)

# ── 检查 1: token 之间的余弦相似度 ──
emb   = style.style_embeds.detach().float()           # (6, 4096)
normed = torch.nn.functional.normalize(emb, dim=1)
sim    = normed @ normed.T                            # (6, 6)

print("\n━━━ Style token 两两余弦相似度 (对角为 1) ━━━")
for row in sim:
    print("  " + "  ".join(f"{v:+.3f}" for v in row.tolist()))

off_diag = sim[~torch.eye(6, dtype=torch.bool)]
print(f"\n  off-diagonal mean : {off_diag.mean():+.4f}")
print(f"  off-diagonal max  : {off_diag.max():+.4f}")
print(f"  off-diagonal min  : {off_diag.min():+.4f}")

# ── 检查 2: 和普通 token embedding 的距离(确保在 manifold 上) ──
rand_ids = torch.randint(1000, 100000, (500,), device=emb.device)
ref      = embed_fn(rand_ids).float()
ref_norm = ref.norm(dim=1).mean().item()
our_norm = emb.norm(dim=1).mean().item()
print(f"\n  style_embeds L2 均值 : {our_norm:.3f}")
print(f"  random  tokens L2 均值: {ref_norm:.3f}")
print(f"  比值 (应在 0.5 ~ 2.0 之间): {our_norm/ref_norm:.3f}")

# ── 检查 3: slot_gate 初始必须是 0 ──
gate = torch.tanh(style.slot_gate).item()
print(f"\n  slot_gate (tanh)     : {gate:+.6f}  (必须 ≈ 0)")

# ── 判定 ──
print("\n" + "=" * 50)
ok = True
if off_diag.mean() > 0.95:
    print("  ❌ FAIL: style token 之间过于相似,init 还有问题")
    ok = False
else:
    print(f"  ✅ token 多样性 OK (mean cos = {off_diag.mean():.3f})")

if not (0.5 < our_norm/ref_norm < 2.0):
    print(f"  ❌ FAIL: style_embeds L2 范数异常 ({our_norm/ref_norm:.3f})")
    ok = False
else:
    print(f"  ✅ L2 范数在合理区间")

if abs(gate) > 1e-4:
    print(f"  ❌ FAIL: slot_gate 初始不为 0")
    ok = False
else:
    print(f"  ✅ slot_gate 初始为 0")

print("=" * 50)
print("可以开训练" if ok else "先修 style_prefix.py 再训练")