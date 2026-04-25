"""
诊断脚本: 查清楚 Qwen3-VL visual 的内部结构,
找出为什么 slot 参数 0 更新。
"""
import sys
sys.path.insert(0, '/opt/data/private/qwen3-vl-master/qwen3-vl')

import torch
import inspect
from qwenvl.model.rs_caption_model import RSCaptionModel
from qwenvl.model.semantic_slot import SemanticSlotModule

MP = "/opt/data/private/qwen3-vl-master/qwen3-vl/pretrained/Qwen3-VL-8B-Instruct"

print("加载模型中(会花 2-3 分钟)...")
model = RSCaptionModel.from_pretrained(
    MP, torch_dtype=torch.bfloat16, device_map="auto"
)

visual = model.model.visual

# ══════════════════════════════════════════════════════════════
# 诊断 1: visual 有哪些子模块? merger 是其中之一还是主力?
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("诊断 1: visual 子模块")
print("=" * 70)
for name, mod in visual.named_children():
    num_params = sum(p.numel() for p in mod.parameters()) / 1e6
    print(f"  {name:30s}  {type(mod).__name__:40s}  {num_params:7.2f}M")

# ══════════════════════════════════════════════════════════════
# 诊断 2: 把 visual.forward 源码打出来
# 关键:看 merger 的返回值到底被怎么用了,是否被后续处理覆盖
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("诊断 2: visual.forward 源码")
print("=" * 70)
try:
    src = inspect.getsource(visual.forward)
    print(src)
except Exception as e:
    print(f"  [ERROR] 无法获取源码: {e}")

# ══════════════════════════════════════════════════════════════
# 诊断 3: merger 本身的 forward 源码
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("诊断 3: merger.forward 源码")
print("=" * 70)
try:
    print(f"类型: {type(visual.merger).__name__}")
    src = inspect.getsource(visual.merger.forward)
    print(src)
except Exception as e:
    print(f"  [ERROR] 无法获取源码: {e}")

# ══════════════════════════════════════════════════════════════
# 诊断 4: deepstack_merger_list 存在吗? 它会不会绕过我们的 hook?
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("诊断 4: deepstack_merger_list")
print("=" * 70)
if hasattr(visual, "deepstack_merger_list"):
    dsml = visual.deepstack_merger_list
    print(f"  存在, 数量={len(dsml)}")
    print(f"  第一个类型: {type(dsml[0]).__name__}")
    n_params = sum(p.numel() for p in dsml.parameters()) / 1e6
    print(f"  总参数量: {n_params:.2f}M")
else:
    print("  不存在")

# ══════════════════════════════════════════════════════════════
# 诊断 5: 实际跑一次 visual forward, 看输出结构
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("诊断 5: 实际 visual forward 输出结构")
print("=" * 70)

# 构造假 input: 1 张 448×448 的图,merge_size=2, patch=14
# → grid_thw = [1, 32, 32], num_patches = 1024, 经 merger 后 = 256 tokens
dummy_pixel = torch.randn(1024, 1176, device="cuda", dtype=torch.bfloat16)  # (num_patches, C*ph*pw)
dummy_grid  = torch.tensor([[1, 32, 32]], device="cuda")

# 给 merger 装个 hook, 看它的输出
merger_outputs = {}
def merger_hook(mod, inp, out):
    merger_outputs["output"] = out
    merger_outputs["output_type"] = type(out).__name__
    if isinstance(out, torch.Tensor):
        merger_outputs["shape"] = list(out.shape)
    elif isinstance(out, (tuple, list)):
        merger_outputs["len"] = len(out)
        merger_outputs["elem_types"] = [type(o).__name__ for o in out]
        merger_outputs["elem_shapes"] = [
            list(o.shape) if isinstance(o, torch.Tensor) else None for o in out
        ]

h = visual.merger.register_forward_hook(merger_hook)

with torch.no_grad():
    try:
        visual_out = visual(dummy_pixel, grid_thw=dummy_grid)
        print(f"  visual() 返回类型: {type(visual_out).__name__}")
        if hasattr(visual_out, "last_hidden_state"):
            print(f"    .last_hidden_state: {list(visual_out.last_hidden_state.shape)}")
        if hasattr(visual_out, "deepstack_features"):
            ds = visual_out.deepstack_features
            if ds is not None:
                if isinstance(ds, (list, tuple)):
                    print(f"    .deepstack_features: list, 长度 {len(ds)}")
                    for i, d in enumerate(ds[:3]):
                        if isinstance(d, torch.Tensor):
                            print(f"      [{i}]: {list(d.shape)}")
                else:
                    print(f"    .deepstack_features: {type(ds).__name__}")
    except Exception as e:
        import traceback
        print(f"  visual forward 失败: {e}")
        traceback.print_exc()
    finally:
        h.remove()

print(f"\n  merger 的 hook 捕获:")
for k, v in merger_outputs.items():
    print(f"    {k}: {v}")

# ══════════════════════════════════════════════════════════════
# 诊断 6: visual.forward 里 merger 后还做了什么?
# 找可能"再次变换"输出的代码行
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("诊断 6: visual.forward 源码中 'merger' / 'deepstack' 的上下文")
print("=" * 70)
try:
    src = inspect.getsource(visual.forward)
    lines = src.split('\n')
    for i, line in enumerate(lines):
        if 'merger' in line.lower() or 'deepstack' in line.lower():
            # 打印这行及上下 2 行
            start = max(0, i - 1)
            end   = min(len(lines), i + 3)
            print(f"  --- 行 {i} ---")
            for j in range(start, end):
                marker = ">>> " if j == i else "    "
                print(f"  {marker}{lines[j]}")
            print()
except Exception as e:
    print(f"  [ERROR]: {e}")

print("\n" + "=" * 70)
print("诊断完成")
print("=" * 70)