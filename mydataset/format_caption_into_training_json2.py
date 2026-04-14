import json
import argparse
import os

# 卫星图根目录
SATELLITE_BASE = "/opt/data/private/qwen3-vl-master/data/university-s-train"
# 无人机图根目录
DRONE_BASE     = "/opt/data/private/qwen3-vl-master/data/university-d-train-selected-5"

def convert(data_root, output_file):
    """
    data_root: 完整的原始 JSON 字典
    output_file: 输出的 jsonl 文件路径
    """
    # 核心修改：从字典中提取 "images" 列表进行遍历
    samples = data_root.get("images", [])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in samples:
            # 修改点 1：根据数据格式，直接访问键名，不再嵌套 ["images"]
            sat_filename = item["satellite_image"]   # 例如 "0922.jpg"
            drone_imgs = item["drone_images"]        # 例如 ["0922/image-07.jpeg", ...]

            # 修改点 2：卫星图路径拼接（数据中 satellite_image 已经是文件名）
            sat_full_path = os.path.join(SATELLITE_BASE, sat_filename.split('/')[-1])
            
            # 无人机图路径拼接
            image_paths = [sat_full_path] + [os.path.join(DRONE_BASE, img) for img in drone_imgs]
            
            # 生成多图标签
            image_tags = "\n".join(["<image>"] * len(image_paths))

            # 修改点 3：获取描述文本。数据中描述在 sentences 列表的第一项 raw 字段
            # 注意：如果想用全局 description，请使用 data_root["description"]
            # 但通常微调需要的是每张图对应的具体描述：
            caption = item["sentences"][0]["raw"]

            qwen_entry = {
                "images": image_paths,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"{image_tags}\nPlease generate a detailed description for these images.",
                    },
                    {
                        "from": "gpt",
                        "value": caption
                    }
                ]
            }
            f.write(json.dumps(qwen_entry, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert caption dataset to Qwen3-VL training format")
    parser.add_argument("--input",  required=True, help="Input JSON file path")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    # 传入整个加载的 json 字典
    convert(input_data, args.output)
    
    # 修正打印总数逻辑
    total_processed = len(input_data.get("images", []))
    print(f"✅ 转换完成！共处理 {total_processed} 条样本，输出至 {args.output}")