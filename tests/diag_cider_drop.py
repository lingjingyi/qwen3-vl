"""
对比 A4 (纯 LoRA) 和 A0 (Full) 的 prediction,
找出 slot+style 具体'丢失'了哪些高 TF-IDF 词汇.
"""
import json
import re
from collections import Counter
from math import log

def load(p):
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)}

a4 = load("/opt/data/private/qwen3-vl-master/qwen3-vl/dataset/pred_A4_pure_lora.jsonl")
a0 = load("/opt/data/private/qwen3-vl-master/qwen3-vl/dataset/pred_A0_repro.jsonl")
ref = load("/opt/data/private/qwen3-vl-master/qwen3-vl/dataset/rsgpt_test.jsonl")

def get_ref(item):
    for t in item.get("conversations", []):
        if t.get("from") == "gpt":
            return t["value"]
    return ""

def tok(text):
    return re.findall(r"\b\w+\b", text.lower())

# ── IDF 用 reference 语料估 ──
N = len(ref)
df = Counter()
for it in ref.values():
    toks = set(tok(get_ref(it)))
    for t in toks:
        df[t] += 1
idf = {t: log(N / (1 + c)) for t, c in df.items()}

# ── 每条样本, 对比 A4 和 A0 的词汇差异 ──
# 统计: 哪些 reference 里的"高 IDF 词", A4 命中但 A0 没命中
a4_wins = Counter()  # A4 命中 A0 不中的词
a0_wins = Counter()  # A0 命中 A4 不中的词

for sid in sorted(set(a4) & set(a0) & set(ref)):
    ref_toks = set(tok(get_ref(ref[sid])))
    a4_toks  = set(tok(a4[sid]["prediction"]))
    a0_toks  = set(tok(a0[sid]["prediction"]))
    for t in ref_toks:
        if t in a4_toks and t not in a0_toks:
            a4_wins[t] += 1
        elif t in a0_toks and t not in a4_toks:
            a0_wins[t] += 1

# ── 按 IDF × 命中次数排序 ──
print("=" * 70)
print("A4(纯LoRA) 比 A0(Full) 多命中的高 IDF 参考词 (Top 30)")
print("这些词加了 slot+style 后反而丢失 → CIDEr 扣分来源")
print("=" * 70)
scored = [(t, c, idf.get(t, 0), c * idf.get(t, 0))
          for t, c in a4_wins.items() if t in idf]
scored.sort(key=lambda x: -x[3])
for t, c, i, s in scored[:30]:
    print(f"  {t:25s}  命中次数 {c:3d}  IDF {i:.2f}  TF-IDF {s:.2f}")

print()
print("=" * 70)
print("A0(Full) 比 A4(纯LoRA) 多命中的参考词 (Top 30)")
print("这些是 slot+style 的正收益")
print("=" * 70)
scored = [(t, c, idf.get(t, 0), c * idf.get(t, 0))
          for t, c in a0_wins.items() if t in idf]
scored.sort(key=lambda x: -x[3])
for t, c, i, s in scored[:30]:
    print(f"  {t:25s}  命中次数 {c:3d}  IDF {i:.2f}  TF-IDF {s:.2f}")

# ── 净影响 ──
net_loss_score = sum(c * idf.get(t, 0) for t, c in a4_wins.items() if t in idf)
net_gain_score = sum(c * idf.get(t, 0) for t, c in a0_wins.items() if t in idf)
print()
print("=" * 70)
print(f"A4 净赢 TF-IDF 分数: {net_loss_score:.1f}  ({len(a4_wins)} 个词)")
print(f"A0 净赢 TF-IDF 分数: {net_gain_score:.1f}  ({len(a0_wins)} 个词)")
print(f"差额: {net_loss_score - net_gain_score:+.1f}  (正数 = slot+style 损失多)")