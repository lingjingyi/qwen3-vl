import json
import os

INPUT_JSON  = "/opt/data/private/qwen3-vl-master/qwen3-vl/dataset/dataset_train2.json"   # 修改为你的输入文件路径
OUTPUT_JSON = "/opt/data/private/qwen3-vl-master/qwen3-vl/dataset/train2.json"  # 输出文件路径
DRONE_BASE  = "/opt/data/private/qwen3-vl-master/data/university-d-train-selected-5"
# ========== 关键修改1：添加卫星图实际基础路径 ==========
SATELLITE_BASE = "/opt/data/private/qwen3-vl-master/data/university-s-train/"

def get_drone_images(item_id: str) -> list[str]:
    """读取 selected_images.txt，返回 'id/filename' 格式的路径列表。"""
    txt_path = os.path.join(DRONE_BASE, item_id, "selected_images.txt")
    if not os.path.isfile(txt_path):
        print(f"  [WARN] 找不到 selected_images.txt：{txt_path}")
        return []

    results = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            filename = line.strip()
            if filename:                          # 跳过空行
                results.append(f"{item_id}/{filename}")
    return results


def build_images_block(item_id: str) -> dict:
    return {
        "original_id":      item_id,
        "split":            "train",
        # ========== 关键修改2：拼接卫星图完整绝对路径 ==========
        "satellite_image":  os.path.join(SATELLITE_BASE, f"{item_id}.jpg"),
        "drone_images":     get_drone_images(item_id),
    }


def insert_after_description(item: dict) -> dict:
    """在 description 键之后插入 images，保持其余键顺序不变。"""
    new_item = {}
    for key, value in item.items():
        new_item[key] = value
        if key == "description":
            new_item["images"] = build_images_block(item["id"])
    # 若原始数据中没有 description 键，则追加到末尾
    if "images" not in new_item:
        new_item["images"] = build_images_block(item["id"])
    return new_item


def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 兼容顶层是列表或带某个列表字段的字典
    if isinstance(data, list):
        records = data
        wrapper = None
    elif isinstance(data, dict):
        # 自动找第一个 list 类型的值作为记录列表
        wrapper_key = next((k for k, v in data.items() if isinstance(v, list)), None)
        if wrapper_key is None:
            raise ValueError("JSON 顶层字典中未找到列表字段")
        records    = data[wrapper_key]
        wrapper    = (data, wrapper_key)
    else:
        raise ValueError("不支持的 JSON 结构")

    updated = []
    for item in records:
        if "id" not in item:
            print(f"  [WARN] 条目缺少 id 字段，已原样保留：{item}")
            updated.append(item)
            continue
        updated.append(insert_after_description(item))
        print(f"  [OK] 已处理 id={item['id']}，drone_images 数量={len(updated[-1]['images']['drone_images'])}")

    if wrapper is None:
        result = updated
    else:
        wrapper[0][wrapper[1]] = updated
        result = wrapper[0]

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完成！共处理 {len(updated)} 条记录，结果已写入 {OUTPUT_JSON}")


if __name__ == "__main__":
    main()