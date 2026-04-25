"""
B1 训练失败诊断: 不重训, 从 checkpoint / predictions 反向分析.
1. 训练 loss 曲线 (过拟合?)
2. style_embeds 最终状态 (学偏了?)
3. slot_gate 值 (slot-style 连接强度)
4. Predictions 质量对比
"""
import json
from pathlib import Path

import torch

BASE = Path("/opt/data/private/qwen3-vl-master/qwen3-vl")
A0_PRED = BASE / "dataset/rsgpt_evaluation_results_A0.json"
B1_PRED = BASE / "dataset/pred_repair_v1.jsonl"        # 你这次的 output
A0_CKPT = BASE / "output/rsgpt_bleu_fix_stage1/checkpoint-730/merger_weights.pt"

# ── 请补全你 B1 训练的 checkpoint 路径 ──
# 找最新 checkpoint
B1_BASE = BASE / "output/rsgpt_b_repair_v1"
candidates = sorted(
    [p for p in B1_BASE.glob("checkpoint-*") if p.is_dir()],
    key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
    reverse=True,
)
if not candidates:
    print("❌ 未找到 B1 checkpoint")
    exit()
B1_CKPT = candidates[0] / "merger_weights.pt"
print(f"B1 checkpoint: {B1_CKPT}")

# ════════════════════════════════════════════════════════════════════
# 1. 训练 loss 曲线 (过拟合检测)
# ════════════════════════════════════════════════════════════════════
trainer_state = candidates[0] / "trainer_state.json"
if trainer_state.exists():
    with open(trainer_state) as f:
        ts = json.load(f)
    log_history = ts.get("log_history", [])
    losses = [(e["step"], e.get("loss", None))
              for e in log_history if "loss" in e]
    print("\n" + "=" * 70)
    print("训练 loss 曲线 (每 50 step 采样)")
    print("=" * 70)
    if losses:
        for step, loss in losses[::5]:  # 每 5 个取一个
            print(f"  step {step:5d}  loss {loss:.4f}")
        first, last = losses[0][1], losses[-1][1]
        print(f"\n  起始 loss: {first:.4f}")
        print(f"  终止 loss: {last:.4f}")
        if last < 0.5:
            print(f"  🔴 末期 loss < 0.5, 可能严重过拟合")
        elif last < 0.8:
            print(f"  ⚠️  末期 loss < 0.8, 轻度过拟合迹象")
        else:
            print(f"  ✅ 末期 loss 在合理范围")
    else:
        print("  (无 loss 数据)")

# ════════════════════════════════════════════════════════════════════
# 2. style_embeds 最终状态对比
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("style_embeds 最终状态对比")
print("=" * 70)

def load_style(p):
    sd = torch.load(p, map_location="cpu", weights_only=True)
    for k, v in sd.items():
        if "style_embeds" in k:
            return v
    return None

a0_s = load_style(A0_CKPT)
b1_s = load_style(B1_CKPT)

if a0_s is not None and b1_s is not None:
    def stats(t, name):
        t = t.float()
        norm = torch.norm(t, dim=1)
        # 两两余弦相似度
        normed = torch.nn.functional.normalize(t, dim=1)
        sim = normed @ normed.T
        off = sim[~torch.eye(t.size(0), dtype=torch.bool)]
        print(f"\n  [{name}]  shape={list(t.shape)}")
        print(f"    L2 norm:    mean={norm.mean():.3f}  range=[{norm.min():.3f}, {norm.max():.3f}]")
        print(f"    两两 cos:    mean={off.mean():+.3f}  max={off.max():+.3f}")
        return norm.mean().item(), off.mean().item()

    a0_n, a0_c = stats(a0_s, "A0 (K=6, BLEU-1=0.5082)")
    b1_n, b1_c = stats(b1_s, "B1 (K=8, BLEU-1=0.5038)")

    print(f"\n  诊断:")
    if b1_c > a0_c + 0.1:
        print(f"    🔴 B1 的 style token 更相似 (cos {b1_c:+.3f} vs {a0_c:+.3f})")
        print(f"       style 可能过度收敛到模板, 失去多样性")
    if b1_n < a0_n * 0.5 or b1_n > a0_n * 2:
        print(f"    🔴 L2 norm 异常 (B1={b1_n:.2f}, A0={a0_n:.2f})")
    if abs(b1_c - a0_c) < 0.05 and abs(b1_n - a0_n) < a0_n * 0.3:
        print(f"    ✅ style_embeds 状态与 A0 相近, 不是 style 本身出问题")

# ════════════════════════════════════════════════════════════════════
# 3. slot_gate 值 (style 吸收了多少 slot 信息?)
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("slot_gate 最终值 (tanh 后在 [-1, 1])")
print("=" * 70)

def load_gate(p):
    sd = torch.load(p, map_location="cpu", weights_only=True)
    for k, v in sd.items():
        if "slot_gate" in k:
            return v.item()
    return None

a0_g = load_gate(A0_CKPT)
b1_g = load_gate(B1_CKPT)

import math
if a0_g is not None and b1_g is not None:
    a0_gt = math.tanh(a0_g)
    b1_gt = math.tanh(b1_g)
    print(f"  A0: gate={a0_g:+.4f}  tanh(gate)={a0_gt:+.4f}")
    print(f"  B1: gate={b1_g:+.4f}  tanh(gate)={b1_gt:+.4f}")
    if abs(b1_gt) < 0.01:
        print(f"  🔴 B1 的 slot_gate 接近 0, slot 对 style 几乎无贡献")
    elif abs(a0_gt - b1_gt) < 0.02:
        print(f"  ✅ B1 和 A0 的 slot_gate 相近, slot-style 连接强度相似")

# ════════════════════════════════════════════════════════════════════
# 4. refine_scale 值
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("refine_scale 最终值 (控制 slot 对视觉的扰动强度)")
print("=" * 70)

def load_rs(p):
    sd = torch.load(p, map_location="cpu", weights_only=True)
    for k, v in sd.items():
        if "refine_scale" in k:
            return v.item()
    return None

a0_rs = load_rs(A0_CKPT)
b1_rs = load_rs(B1_CKPT)
print(f"  A0 (init 0.05, {a0_rs:+.4f})")
print(f"  B1 (init 0.02, {b1_rs:+.4f})")
if abs(b1_rs) < 0.01:
    print(f"  🔴 B1 refine_scale 接近 0, slot 对视觉 token 的修改几乎消失")

# ════════════════════════════════════════════════════════════════════
# 5. Prediction 行为对比
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Prediction 长度与句式对比")
print("=" * 70)

def pred_stats(p):
    with open(p) as f:
        ds = [json.loads(l) for l in f]
    words = [len(d["prediction"].split()) for d in ds]
    starts_with_this = sum(1 for d in ds if d["prediction"].startswith("This is"))
    starts_with_the  = sum(1 for d in ds if d["prediction"].startswith("The image"))
    return {
        "n":     len(ds),
        "words": sum(words) / len(words),
        "this":  starts_with_this,
        "the":   starts_with_the,
    }

a0_stats = pred_stats(A0_PRED)
b1_stats = pred_stats(B1_PRED)

print(f"  A0: n={a0_stats['n']} avg_words={a0_stats['words']:.1f} "
      f"'This is' 开头={a0_stats['this']} 'The image' 开头={a0_stats['the']}")
print(f"  B1: n={b1_stats['n']} avg_words={b1_stats['words']:.1f} "
      f"'This is' 开头={b1_stats['this']} 'The image' 开头={b1_stats['the']}")

if b1_stats['this'] < a0_stats['this'] - 10:
    print(f"  🔴 B1 用 'This is' 开头显著减少, style 锚定失效")
elif b1_stats['this'] > a0_stats['this']:
    print(f"  ⚠️  B1 更多用 'This is' 开头, 但 BLEU 反降, 说明不是风格问题")

# 找 5 个 ID, 对比 A0 和 B1 的预测
a0_d = {json.loads(l)['id']: json.loads(l)['prediction'] for l in open(A0_PRED)}
b1_d = {json.loads(l)['id']: json.loads(l)['prediction'] for l in open(B1_PRED)}
common = sorted(set(a0_d) & set(b1_d))[:5]

print("\n" + "=" * 70)
print("5 条样例对比")
print("=" * 70)
for cid in common:
    print(f"\n[{cid}]")
    print(f"  A0 ({len(a0_d[cid].split())}w): {a0_d[cid][:180]}...")
    print(f"  B1 ({len(b1_d[cid].split())}w): {b1_d[cid][:180]}...")

print("\n" + "█" * 70)
print("诊断完成")
print("█" * 70)