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
    """扫描卫星图像目录，返回 {id: jpg_path}"""
    satellite_dir = Path(satellite_dir)
    satellite_images = {}
    
    for jpg_file in sorted(satellite_dir.glob("*.jpg")):
        img_id = jpg_file.stem  # 文件名不带扩展名
        satellite_images[img_id] = str(jpg_file)
    
    return satellite_images


def scan_drone_images(drone_dir: str):
    """扫描无人机图像目录，返回 {id: [image_paths]}"""
    drone_dir = Path(drone_dir)
    drone_images = defaultdict(list)
    
    for subdir in sorted(drone_dir.iterdir()):
        if subdir.is_dir():
            img_id = subdir.name
            for img_file in sorted(subdir.glob("*.jpeg")):
                drone_images[img_id].append(str(img_file))
    
    return dict(drone_images)


def load_captions_from_dataset_json(dataset_json: str):
    """从 dataset.json 加载描述文本"""
    with open(dataset_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    captions = {}
    items = data if isinstance(data, list) else []
    
    for item in items:
        # 获取 ID 并格式化为 4 位字符串 (例如 "1" -> "0001")
        raw_id = item.get("id")
        if raw_id is None:
            continue
            
        clean_id = str(raw_id).strip().zfill(4)
        
        # 获取描述文本 (对应你 JSON 中的 description 字段)
        description = item.get("description", "").strip()
        
        if description:
            captions[clean_id] = description
    
    return captions


def convert_to_qwen3vl_format(
    satellite_dir: str,
    drone_dir: str,
    caption_file: str,
    output_file: str,
    prompt: str = "Please describe the appearance, structure, and surrounding environment of these buildings.",#请描述这些建筑的外观、结构和周边环境。
    include_satellite: bool = True,
    min_drone_images: int = 1,
    max_drone_images: int = None,
):
    """
    转换数据集为 Qwen3-VL 训练格式
    
    Args:
        satellite_dir: 卫星图像目录
        drone_dir: 无人机图像目录
        caption_file: 描述文本 JSON 文件
        output_file: 输出 JSONL 文件
        prompt: 提示词
        include_satellite: 是否包含卫星图像
        min_drone_images: 最少无人机图像数量
        max_drone_images: 最多无人机图像数量 (None 表示不限制)
    """
    
    print("扫描卫星图像...")
    satellite_images = scan_satellite_images(satellite_dir)
    print(f"  找到 {len(satellite_images)} 张卫星图像")
    
    print("扫描无人机图像...")
    drone_images = scan_drone_images(drone_dir)
    print(f"  找到 {len(drone_images)} 个场景的无人机图像")
    
    print("加载描述文本...")
    captions = load_captions_from_dataset_json(caption_file)
    print(f"  找到 {len(captions)} 条描述文本")
    
    # 找到共同存在的场景
    common_ids = set(satellite_images.keys()) & set(drone_images.keys())
    print(f"\n共同存在的场景数: {len(common_ids)}")
    
    results = []
    skipped_no_caption = 0
    skipped_no_drone = 0
    
    for img_id in sorted(common_ids):
        sat_img = satellite_images.get(img_id)
        drone_imgs = drone_images.get(img_id, [])
        caption = captions.get(img_id, "")
        
        # 过滤无人机图像数量
        if len(drone_imgs) < min_drone_images:
            skipped_no_drone += 1
            continue
        
        if max_drone_images:
            drone_imgs = drone_imgs[:max_drone_images]
        
        if not caption:
            skipped_no_caption += 1
            continue
        
        # 构建图像列表
        images = []
        if include_satellite and sat_img:
            images.append(sat_img)
        images.extend(drone_imgs)
        
        # 构建对话
        image_tags = "\n".join(["<image>"] * len(images))
        human_value = f"{image_tags}\n{prompt}"
        
        qwen_entry = {
            "id": img_id,
            "images": images,
            "conversations": [
                {"from": "human", "value": human_value},
                {"from": "gpt", "value": caption}
            ]
        }
        
        results.append(qwen_entry)
    
    # 输出文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\n" + "=" * 60)
    print("转换完成!")
    print(f"输出文件: {output_file}")
    print(f"有效样本数: {len(results)}")
    print(f"跳过 (无描述): {skipped_no_caption}")
    print(f"跳过 (无人机图像不足): {skipped_no_drone}")
    
    if results:
        sample = results[0]
        print(f"\n示例样本:")
        print(f"  ID: {sample['id']}")
        print(f"  图像数量: {len(sample['images'])}")
        print(f"  描述长度: {len(sample['conversations'][1]['value'])} 字符")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="转换 University Cross-View 数据集")
    parser.add_argument(
        "--satellite_dir",
        type=str,
        default="/opt/data/private/qwen3-vl-master/data/university-s-train",
        help="Directory containing satellite images"#卫星图像目录
    )
    parser.add_argument(
        "--drone_dir",
        type=str,
        default="/opt/data/private/qwen3-vl-master/data/university-d-train-selected-5",
        help="Directory containing drone images"#无人机图像目录
    )
    parser.add_argument(
        "--caption_file",
        type=str,
        default="/opt/data/private/qwen3-vl-master/qwen3-vl/dataset/output.json",
        help="JSON file containing image captions"#描述文本 JSON 文件
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/opt/data/private/qwen3-vl-master/qwen3-vl/dataset/qwen3vl_train.jsonl",
        help="Output JSONL file"#输出 JSONL 文件(生成测试的jsonl文件)
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Please describe the appearance, structure, and surrounding environment of these buildings.",
        help="The prompt used for the human message"
    )
    parser.add_argument(
        "--no_satellite",
        action="store_true",
        help="Do not include satellite images in the output"#不包含卫星图像
    )
    parser.add_argument(
        "--min_drone",
        type=int,
        default=1,
        help="Minimum number of drone images required per scene"#最少无人机图像数量
    )
    parser.add_argument(
        "--max_drone",
        type=int,
        default=None,
        help="Maximum number of drone images to include per scene"#最多无人机图像数量
    )
    
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