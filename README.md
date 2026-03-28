# Qwen3-VL 微调框架

基于 RSCC 项目的 Qwen-VL 微调框架，适配 Qwen3-VL 模型用于 Caption 生成任务。

## 目录结构

```
qwen3-vl/
├── qwenvl/
│   ├── train/
│   │   ├── argument.py       # 参数配置
│   │   ├── trainer.py        # 自定义 Trainer
│   │   └── train_qwen.py     # 主训练入口
│   └── data/
│       ├── __init__.py       # 数据集注册
│       ├── data_qwen.py      # 数据处理
│       └── rope2d.py         # RoPE 实现
├── inference/
│   ├── __init__.py
│   └── generate_caption.py   # Caption 生成脚本
├── evaluation/
│   ├── __init__.py
│   └── evaluate_caption.py   # Caption 评估脚本
├── scripts/
│   └── sft.sh                # 训练启动脚本
├── mydataset/
│   └── format_caption_into_training_json.py  # 数据转换脚本
└── README.md                 # 本文件
```

## 环境要求

```bash
conda env create -f ../environment_qwenvl_ft.yaml
conda activate qwenvl_ft
```

主要依赖：
- torch==2.6.0
- transformers>=4.50.0
- deepspeed==0.16.4
- flash_attn==2.7.4.post1

## 快速开始

### 1. 准备数据

将你的数据转换为 Qwen3-VL 训练格式：

```json
{
  "image": "path/to/image.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n请为这张图片生成一个详细的描述。"
    },
    {
      "from": "gpt",
      "value": "这是一张关于...的图片"
    }
  ]
}
```

多图格式（如变化描述）：
```json
{
  "images": ["pre.jpg", "post.jpg"],
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n<image>\n描述两张图片的变化。"
    },
    {
      "from": "gpt",
      "value": "变化描述..."
    }
  ]
}
```

使用数据转换脚本：
```bash
python mydataset/format_caption_into_training_json.py \
    --input your_data.json \
    --output your_data.jsonl
```
python qwen3-vl/mydataset/format_caption_into_training_json.py \
    --input /opt/data/private/qwen3-vl-master/qwen3-vl/dataset/output.json \
    --output /opt/data/private/qwen3-vl-master/qwen3-vl/dataset/dataset_train.jsonl

### 2. 注册数据集

编辑 `qwenvl/data/__init__.py`：

```python
YOUR_CAPTION_DATASET = {
    "annotation_path": "/path/to/your_captions.jsonl",
    "data_path": "/path/to/images",  # 如果JSON中是相对路径
}

data_dict = {
    "your_caption_dataset": YOUR_CAPTION_DATASET,
}
```

### 3. 配置训练参数

编辑 `scripts/sft.sh`：
- `MODEL_PATH`: Qwen3-VL 模型路径
- `DATASETS`: 数据集名称（如 `your_caption_dataset%100`）
- `OUTPUT_DIR`: 输出目录
- 调整学习率、batch size 等超参数

### 4. 启动训练

```bash
cd qwen3-vl
chmod +x scripts/sft.sh
bash scripts/sft.sh
```

## 关键参数说明

### 模型微调控制

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `tune_mm_llm` | 是否微调 LLM 主体 | `True` |
| `tune_mm_vision` | 是否微调视觉编码器 | `False` (节省显存) |
| `tune_mm_mlp` | 是否微调视觉投影层 | `True` |

### 学习率

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `learning_rate` | LLM 学习率 | `1e-6 ~ 2e-6` |
| `mm_projector_lr` | 投影层学习率 | `1e-5` |
| `vision_tower_lr` | 视觉编码器学习率 | `1e-6` |

### 图像分辨率

| 参数 | 说明 |
|------|------|
| `max_pixels` | 最大像素数，格式 `H*W*3` |
| `min_pixels` | 最小像素数 |

推荐值：
- 448x448: `--max_pixels $((448*448*3))`
- 512x512: `--max_pixels $((512*512*3))`

## 推理

### 使用脚本批量生成 Caption

准备测试数据格式 (`test_data.json`):
```json
[
    {"id": "001", "image": "test1.jpg"},
    {"id": "002", "image": "test2.jpg"},
    {"id": "003", "images": ["pre3.jpg", "post3.jpg"]}
]
```

运行批量生成：
```bash
python inference/generate_caption.py \
    --model_path ./checkpoints/qwen3vl_caption \
    --input_file test_data.json \
    --output_file predictions.jsonl \
    --image_key image \
    --id_key id \
    --prompt "请为这张图片生成一个详细的描述。" \
    --max_new_tokens 512
```

cd /opt/data/private/qwen3-vl-master/qwen3-vl/  # 回到项目根目录
python inference/generate_caption.py \
    --model_path /opt/data/private/qwen3-vl-master/qwen3-vl/checkpoints/qwen3vl_caption/merged_model \
    --input_file /opt/data/private/qwen3-vl-master/qwen3-vl/dataset/qwen3vl_train.jsonl \
    --output_file /opt/data/private/qwen3-vl-master/qwen3-vl/dataset/predictions.jsonl \
    --image_key images \
    --id_key id \
    --prompt "Please generate a detailed description for this picture." \
    --max_new_tokens 512

### 单张图片推理代码

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "./checkpoints/qwen3vl_caption",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("./checkpoints/qwen3vl_caption")

messages = [
    {"role": "user", "content": [
        {"type": "image", "image": "test.jpg"},
        {"type": "text", "text": "请描述这张图片。"}
    ]}
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, images=["test.jpg"], return_tensors="pt").to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
```

## 评估

### 准备参考数据

参考数据格式 (`reference.json`):
```json
[
    {"id": "001", "reference": "这是一张关于...的图片"},
    {"id": "002", "reference": "这是另一张...的图片"}
]
```

### 运行评估

```bash
python evaluation/evaluate_caption.py \
    --prediction_file predictions.jsonl \
    --reference_file reference.json \
    --output_file evaluation_results.json \
    --prediction_key prediction \
    --reference_key reference \
    --id_key id
```
python evaluation/evaluate_caption.py \
    --prediction_file /opt/data/private/qwen3-vl-master/qwen3-vl/dataset/predictions.jsonl \
    --reference_file /opt/data/private/qwen3-vl-master/qwen3-vl/dataset/qwen3vl_train.jsonl \
    --prediction_key prediction \
    --reference_key conversations \
    --id_key id \
    --t5_model_path /opt/data/private/qwen3-vl-master/qwen3-vl/pretrained/sentence-t5-base \
    --bert_model_path /opt/data/private/qwen3-vl-master/qwen3-vl/pretrained/roberta-large \
    --output_file /opt/data/private/qwen3-vl-master/qwen3-vl/dataset/evaluation_results.json
    
缺失库需要安装：pip install nltk rouge_score sentence-transformers bert_score -i https://pypi.tuna.tsinghua.edu.cn/simple

### 评估指标

| 指标 | 说明 |
|------|------|
| BLEU-1/2/3/4 | N-gram 匹配度 |
| ROUGE-1/2/L | 基于重叠的文本相似度 |
| METEOR | 词对齐相似度 |
| BERTScore | 语义相似度 |
| T5 Cosine Similarity | Sentence-T5 嵌入余弦相似度 |
| T5 Cubed Similarity | 三次方相似度 |

### 直接在 Python 中使用

```python
from evaluation.evaluate_caption import CaptionEvaluator

evaluator = CaptionEvaluator()

predictions = ["生成的caption1", "生成的caption2"]
references = ["参考caption1", "参考caption2"]

results = evaluator.evaluate(predictions, references)
print(results)
```

## 常见问题

### Q: 显存不足怎么办？
A: 
1. 设置 `tune_mm_vision=False`
2. 减小 `per_device_train_batch_size`
3. 增大 `gradient_accumulation_steps`
4. 启用 `gradient_checkpointing=True`
5. 使用 DeepSpeed Zero-3

### Q: Qwen3-VL 模型在哪里下载？
A: 可以从 HuggingFace 下载：`Qwen/Qwen3-VL-7B-Instruct`

### Q: 如何使用多 GPU 训练？
A: 脚本会自动检测可用的 GPU 数量，使用 `torchrun` 启动分布式训练。

## 参考

- 原项目: [RSCC](https://github.com/Bili-Sakura/RSCC)
- Qwen 官方: [QwenLM](https://github.com/QwenLM)
