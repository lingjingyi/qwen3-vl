"""
University Cross-View 数据集转换脚本

数据结构:
- 卫星图像: /home/wangcheng/data/unversity-big-after-without-negative/university-s-train/
  - 0001.jpg, 0001.json, 0005.jpg, 0005.json, ...
  
- 无人机图像: /home/wangcheng/data/unversity-big-after-without-negative/university-d-train-selected-5/
  - 0001/image-11.jpeg, 0001/image-15.jpeg, ...
  - 0005/image-19.jpeg, ...
  
- 标注文件: dataset/dataset.json (包含描述文本)

输出格式 (JSONL):
{
  "id": "0001",
  "images": ["satellite/0001.jpg", "drone/0001/image-11.jpeg", ...],
  "conversations": [
    {"from": "human", "value": "<image>\n<image>\n...\n请描述这些建筑。"},
    {"from": "gpt", "value": "描述文本..."}
  ]
}
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

def scan_satellite_images(satellite_dir: str):
    """扫描卫星图像目录，返回 {标准化ID: jpg路径}"""
    satellite_dir = Path(satellite_dir)
    satellite_images = {}
    if not satellite_dir.exists():
        print(f"⚠️ 警告: 卫星图目录不存在: {satellite_dir}")
        return {}
    
    for jpg_file in sorted(satellite_dir.glob("*.jpg")):
        # 核心修改：强制补齐 4 位 ID，确保 432 匹配 0432
        img_id = jpg_file.stem.strip().zfill(4)
        satellite_images[img_id] = str(jpg_file)
    return satellite_images

def scan_drone_images(drone_dir: str):
    """扫描无人机图像目录，返回 {标准化ID: [图片路径列表]}"""
    drone_dir = Path(drone_dir)
    drone_images = defaultdict(list)
    if not drone_dir.exists():
        print(f"⚠️ 警告: 无人机图目录不存在: {drone_dir}")
        return {}
    
    for subdir in sorted(drone_dir.iterdir()):
        if subdir.is_dir():
            # 核心修改：文件夹名也补齐 4 位 ID
            img_id = subdir.name.strip().zfill(4)
            for img_file in sorted(subdir.glob("*.jpeg")):
                drone_images[img_id].append(str(img_file))
    return dict(drone_images)

def load_captions_from_dataset_json(dataset_json: str):
    """适配 University 2.0 格式解析标注"""
    if not os.path.exists(dataset_json):
        print(f"⚠️ 警告: 标注文件不存在: {dataset_json}")
        return {}

    with open(dataset_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    captions = {}
    # 数据在 "images" 键下
    items = data.get("images", []) if isinstance(data, dict) else []
    
    for item in items:
        # 优先取 original_id，没有则取 imgid
        raw_id = item.get("original_id") or item.get("imgid")
        if raw_id is None:
            continue
            
        # 核心修改：标注 ID 也统一补齐 4 位
        clean_id = str(raw_id).strip().zfill(4)
        
        # 提取句子
        if "sentences" in item and len(item["sentences"]) > 0:
            description = item["sentences"][0].get("raw", "").strip()
            if description:
                captions[clean_id] = description
    
    return captions

def convert_to_qwen3vl_format(
    satellite_dir: str,
    drone_dir: str,
    caption_file: str,
    output_file: str,
    prompt: str,
    include_satellite: bool,
    min_drone_images: int,
    max_drone_images: int,
):
    print(f"1. 加载标注文件: {caption_file}")
    captions = load_captions_from_dataset_json(caption_file)
    print(f"   -> 成功解析 {len(captions)} 条描述文本")

    print(f"2. 扫描卫星图库: {satellite_dir}")
    satellite_images = scan_satellite_images(satellite_dir)
    print(f"   -> 找到 {len(satellite_images)} 张卫星图像")
    
    print(f"3. 扫描无人机图库: {drone_dir}")
    drone_images = scan_drone_images(drone_dir)
    print(f"   -> 找到 {len(drone_images)} 个场景的无人机图像")
    
    # 调试信息：打印前三个 ID，确认格式是否统一
    print("\n[ID 匹配性检查]")
    if captions: print(f" - 标注 ID 示例: {list(captions.keys())[:3]}")
    if satellite_images: print(f" - 卫星 ID 示例: {list(satellite_images.keys())[:3]}")
    if drone_images: print(f" - 无人机 ID 示例: {list(drone_images.keys())[:3]}")

    # 三方求交集
    common_ids = sorted(list(set(satellite_images.keys()) & set(drone_images.keys()) & set(captions.keys())))
    print(f"\n>>> 成功匹配的三方共有场景数: {len(common_ids)}")
    
    if len(common_ids) == 0:
        print("❌ 错误：匹配场景数为 0，请检查上方打印的 ID 示例是否格式统一！")
        return

    results = []
    skipped_drone = 0
    
    for img_id in common_ids:
        sat_img = satellite_images[img_id]
        drone_list = drone_images[img_id]
        caption = captions[img_id]
        
        # 过滤无人机图像数量
        if len(drone_list) < min_drone_images:
            skipped_drone += 1
            continue
        
        if max_drone_images:
            drone_list = drone_list[:max_drone_images]
        
        # 组装图像路径列表
        final_images = []
        if include_satellite:
            final_images.append(sat_img)
        final_images.extend(drone_list)
        
        # 构造对话标签
        image_tags = "\n".join(["<image>"] * len(final_images))
        
        qwen_entry = {
            "id": img_id,
            "images": final_images,
            "conversations": [
                {
                    "from": "human", 
                    "value": f"{image_tags}\n{prompt}"
                },
                {
                    "from": "gpt", 
                    "value": caption
                }
            ]
        }
        results.append(qwen_entry)
    
    # 输出 JSONL
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print("\n" + "="*60)
    print(f"转换任务圆满结束!")
    print(f"保存至: {output_file}")
    print(f"有效转换条数: {len(results)}")
    print(f"因无人机图像不足跳过: {skipped_drone} 条")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="适配 University 2.0 格式的 Qwen3-VL 预处理脚本")

    # 路径默认值
    parser.add_argument("--satellite_dir", type=str, default="/opt/data/private/qwen3-vl-master/data/university-s-test")
    parser.add_argument("--drone_dir", type=str, default="/opt/data/private/qwen3-vl-master/data/university-d-test-selected-5")
    parser.add_argument("--caption_file", type=str, default="/opt/data/private/qwen3-vl-master/qwen3-vl/dataset/test_dataset.json")
    parser.add_argument("--output", type=str, default="/opt/data/private/qwen3-vl-master/qwen3-vl/dataset/test_dataset.jsonl")
    
    # 逻辑参数默认值
    parser.add_argument("--prompt", type=str, default="Please describe the appearance, structure, and surrounding environment of these buildings.")
    parser.add_argument("--no_satellite", action="store_true")
    parser.add_argument("--min_drone", type=int, default=1)
    parser.add_argument("--max_drone", type=int, default=5)

    args = parser.parse_args()

    convert_to_qwen3vl_format(
        satellite_dir=args.satellite_dir,
        drone_dir=args.drone_dir,
        caption_file=args.caption_file,
        output_file=args.output,
        prompt=args.prompt,
        include_satellite=not args.no_satellite,
        min_drone_images=args.min_drone,
        max_drone_images=args.max_drone,
    )

if __name__ == "__main__":
    main()