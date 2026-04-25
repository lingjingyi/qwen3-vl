import re

"""
数据集配置文件

配置说明:
- annotation_path: 标注文件的路径 (JSON 或 JSONL 格式)
- data_path: 图像文件的根目录路径 (如果标注文件中使用绝对路径，可以留空)
"""

UNIVERSITY_CROSSVIEW_DATASET = {
    "annotation_path": "/opt/data/private/qwen3-vl-master/qwen3-vl/dataset/train_dataset.jsonl",
    "data_path": "",  # 标注文件中已使用绝对路径
}

# RSGPT 图像描述数据集
# 使用前请先运行 convert_rsgpt.py 生成 rsgpt_train.jsonl
RSGPT_CAPTION_DATASET = {
    "annotation_path": "/opt/data/private/qwen3-vl-master/qwen3-vl/dataset/rsgpt_train.jsonl",
    "data_path": "",  # convert_rsgpt.py 已写入绝对路径
}

RSGPT_CAPTION_DATASET_ALIGNED = {
    "annotation_path": "/opt/data/private/qwen3-vl-master/qwen3-vl/dataset/rsgpt_train_aligned.jsonl",  # 指向重写后的文件
    "data_path": "",  # 和原来一样
}

data_dict = {
    "train_dataset": UNIVERSITY_CROSSVIEW_DATASET,
    "university_crossview": UNIVERSITY_CROSSVIEW_DATASET,
    "rsgpt_train": RSGPT_CAPTION_DATASET,
    "rsgpt_train_aligned": RSGPT_CAPTION_DATASET_ALIGNED,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name  = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict:
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(
                f"Dataset '{dataset_name}' not found. "
                f"Available: {list(data_dict.keys())}"
            )
    return config_list


if __name__ == "__main__":
    import os

    # 可改为 "rsgpt_caption%100" 单独验证 RSGPT，或混合使用
    dataset_names = ["university_crossview%100", "rsgpt_caption%100, rsgpt_train_aligned%100"]
    configs = data_list(dataset_names)

    for config in configs:
        print(config)
        if not os.path.exists(config["annotation_path"]):
            raise FileNotFoundError(
                f"Annotation file not found: {config['annotation_path']}"
            )
        else:
            print(f"  ✅ 文件存在: {config['annotation_path']}")
