import json
import argparse

# 卫星图根目录
SATELLITE_BASE = "/opt/data/private/qwen3-vl-master/data/university-s-train"
# 无人机图根目录
DRONE_BASE     = "/opt/data/private/qwen3-vl-master/data/university-d-train-selected-5"


def convert(input_data, output_file):
    """卫星图（首位）+ 无人机图混合多图模式 + description"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in input_data:
            sat_img    = item["images"]["satellite_image"]   # 完整路径，如 /opt/.../0001.jpg
            drone_imgs = item["images"]["drone_images"]      # 相对路径，如 ["0001/image-11.jpeg", ...]

            # 卫星图：取文件名后重新拼接，避免 sat_img 已含完整路径时重复
            sat_filename = sat_img.split('/')[-1]
            image_paths = (
                [f"{SATELLITE_BASE}/{sat_filename}"] +
                [f"{DRONE_BASE}/{img}" for img in drone_imgs]
            )
            image_tags = "\n".join(["<image>"] * len(image_paths))

            qwen_entry = {
                "images": image_paths,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"{image_tags}\nPlease generate a detailed description for these images.",
                    },
                    {
                        "from": "gpt",
                        "value": item["description"]
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

    convert(input_data, args.output)
    print(f"✅ 转换完成！共处理 {len(input_data)} 条，输出至 {args.output}")