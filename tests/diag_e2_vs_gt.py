"""
tests/diag_e2_vs_gt.py

诊断 E2 (开 count_head) 与 GT 的差距,
找出 E2 在哪些方面还有提升空间, 指导下一步优化方向.
"""
import json
import re
from collections import Counter
from math import log
from pathlib import Path

# ★ 把这里改成你的 E2 文件名
E2_PRED = "/opt/data/private/qwen3-vl-master/qwen3-vl/dataset/pred_count.jsonl"  # ← 改成实际文件名
GT      = "/opt/data/private/qwen3-vl-master/qwen3-vl/dataset/rsgpt_test.jsonl"


def load(p):
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)}


def get_ref(item):
    for t in item.get("conversations", []):
        if t.get("from") == "gpt":
            return t["value"]
    return ""


def tok(t):
    return re.findall(r"\b\w+\b", t.lower())


# ── 加载数据 ──
gt = load(GT)
e2 = load(E2_PRED)
common = sorted(set(gt) & set(e2))
print(f"加载 {len(common)} 个共同样本\n")

# ── 计算 IDF (基于 GT) ──
N  = len(gt)
df = Counter()
for v in gt.values():
    for w in set(tok(get_ref(v))):
        df[w] += 1
idf = {w: log(N / (1 + c)) for w, c in df.items()}

# ════════════════════════════════════════════════════════════════
# 分析 1: 长度差距
# ════════════════════════════════════════════════════════════════
print("=" * 70)
print("分析 1: 长度对比 (是否预测够长)")
print("=" * 70)

gt_lens = [len(get_ref(gt[i]).split()) for i in common]
e2_lens = [len(e2[i]["prediction"].split()) for i in common]

print(f"  GT 平均长度: {sum(gt_lens) / len(gt_lens):.1f} 词")
print(f"  E2 平均长度: {sum(e2_lens) / len(e2_lens):.1f} 词")
print(f"  E2/GT 比例: {sum(e2_lens) / sum(gt_lens) * 100:.1f}%")

shorter = sum(1 for i, sid in enumerate(common) if e2_lens[i] < gt_lens[i] * 0.6)
print(f"  E2 比 GT 短 40%+ 的样本: {shorter}/{len(common)}")
if shorter > len(common) * 0.5:
    print(f"  🔴 超过半数样本明显偏短 → 长度是瓶颈, 推荐 G1 length conditioning")

# ════════════════════════════════════════════════════════════════
# 分析 2: GT 关键词被 E2 丢失 (按 IDF 排序)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("分析 2: GT 中 E2 没说出来的高 IDF 词 (Top 30)")
print("这些词代表 E2 在哪些'特异性描述'上还差")
print("=" * 70)

missed = Counter()
for sid in common:
    gt_words = set(tok(get_ref(gt[sid])))
    e2_words = set(tok(e2[sid]["prediction"]))
    for w in gt_words - e2_words:
        if idf.get(w, 0) > 0.5:    # 只关心稀有词
            missed[w] += 1

scored = [(w, c, idf.get(w, 0), c * idf.get(w, 0)) for w, c in missed.items()]
scored.sort(key=lambda x: -x[3])

for w, c, i, s in scored[:30]:
    print(f"  {w:25s}  GT 出现 {c:3d}次  IDF {i:5.2f}  权重 {s:6.1f}")

# 词性分类提示
nums = [s[0] for s in scored[:30] if s[0] in
        {"two","three","four","five","six","seven","eight","nine","ten",
         "eleven","twelve","fourteen","seventeen","twenty","fifteen","sixteen"}]
colors = [s[0] for s in scored[:30] if s[0] in
          {"white","black","red","blue","green","yellow","brown","gray","grey","orange"}]
spatial = [s[0] for s in scored[:30] if s[0] in
           {"east","west","north","south","corner","top","bottom","left","right",
            "center","middle","upper","lower"}]

if nums:
    print(f"\n  🎯 数字类丢失 ({len(nums)}): {nums}")
    print(f"     → count_head 还没完全解决数字问题")
if colors:
    print(f"  🎯 颜色类丢失 ({len(colors)}): {colors}")
if spatial:
    print(f"  🎯 方位类丢失 ({len(spatial)}): {spatial}")

# ════════════════════════════════════════════════════════════════
# 分析 3: E2 说了但 GT 里没有的词 (产生噪声)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("分析 3: E2 prediction 里 GT 没说的高 IDF 词 (Top 20)")
print("这些是 'hallucination' 或 'over-description'")
print("=" * 70)

extra = Counter()
for sid in common:
    gt_words = set(tok(get_ref(gt[sid])))
    e2_words = set(tok(e2[sid]["prediction"]))
    for w in e2_words - gt_words:
        if idf.get(w, 0) > 0.5:
            extra[w] += 1

scored2 = [(w, c, idf.get(w, 0), c * idf.get(w, 0)) for w, c in extra.items()]
scored2.sort(key=lambda x: -x[3])

for w, c, i, s in scored2[:20]:
    print(f"  {w:25s}  E2 多说 {c:3d}次  IDF {i:5.2f}  权重 {s:6.1f}")

# ════════════════════════════════════════════════════════════════
# 分析 4: 数字精确度
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("分析 4: 数字精确度")
print("=" * 70)

NUMS = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,
        "nine":9,"ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,
        "fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19,"twenty":20}

def extract_nums(text):
    nums = []
    for w in tok(text):
        if w in NUMS:
            nums.append(NUMS[w])
        elif w.isdigit() and 1 <= int(w) <= 50:
            nums.append(int(w))
    return nums

gt_has_num = 0
e2_has_num = 0
both_have  = 0
exact_match = 0
for sid in common:
    g_nums = extract_nums(get_ref(gt[sid]))
    e_nums = extract_nums(e2[sid]["prediction"])
    if g_nums:
        gt_has_num += 1
    if e_nums:
        e2_has_num += 1
    if g_nums and e_nums:
        both_have += 1
        if any(n in e_nums for n in g_nums):
            exact_match += 1

print(f"  GT 含数字的样本: {gt_has_num}/{len(common)}")
print(f"  E2 含数字的样本: {e2_has_num}/{len(common)}")
print(f"  GT 和 E2 都含数字: {both_have}")
if both_have > 0:
    print(f"  数字精确命中: {exact_match}/{both_have} ({100*exact_match/both_have:.1f}%)")
    if exact_match / both_have < 0.4:
        print(f"  🔴 数字精确度低 (<40%) — count_head 学了量但没学准")

# ════════════════════════════════════════════════════════════════
# 分析 5: 看 3 个具体的低分样本
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("分析 5: 3 个 E2 与 GT 差距大的样本")
print("=" * 70)

per_sample = []
for sid in common:
    g_words = set(tok(get_ref(gt[sid])))
    e_words = set(tok(e2[sid]["prediction"]))
    hit = g_words & e_words
    score = sum(idf.get(w, 0) for w in hit)
    total_possible = sum(idf.get(w, 0) for w in g_words)
    per_sample.append((sid, score, total_possible, score / max(total_possible, 1)))

per_sample.sort(key=lambda x: x[3])
for sid, score, total, ratio in per_sample[:3]:
    print(f"\n--- {sid} (命中比例 {ratio*100:.1f}%) ---")
    print(f"GT: {get_ref(gt[sid])[:300]}")
    print(f"E2: {e2[sid]['prediction'][:300]}")

# ════════════════════════════════════════════════════════════════
# 终极建议
# ════════════════════════════════════════════════════════════════
print("\n" + "█" * 70)
print("综合诊断结论")
print("█" * 70)

avg_ratio = sum(s[3] for s in per_sample) / len(per_sample)
print(f"\n  平均高 IDF 词命中率: {avg_ratio*100:.1f}%")
print(f"  长度比例: {sum(e2_lens) / sum(gt_lens) * 100:.1f}%")
print(f"  数字精确度: {100*exact_match/max(both_have,1):.1f}%")

print("\n  下一步推荐:")
if shorter > len(common) * 0.5:
    print("  ★ 长度是首要瓶颈 → G1 Length Conditioning")
elif (100*exact_match/max(both_have,1)) < 40:
    print("  ★ 数字精度是瓶颈 → 改进 count_head (加监督) 或 F1 Multi-scale")
elif spatial and len(spatial) > 5:
    print("  ★ 方位描述是瓶颈 → F2 Slot Diversity Loss")
else:
    print("  ★ 综合瓶颈, 推荐 F1 Multi-scale (覆盖最广)")