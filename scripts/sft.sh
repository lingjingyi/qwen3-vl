#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${PROJECT_DIR}/logs/train_${TIMESTAMP}.log"

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20000-29999 -n 1)}
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)

MODEL_PATH="/opt/data/private/qwen3-vl-master/qwen3-vl/pretrained/Qwen3-VL-8B-Instruct"
OUTPUT_DIR="${PROJECT_DIR}/checkpoints/qwen3vl_caption"
CACHE_DIR="${PROJECT_DIR}/cache"
DS_CONFIG="${SCRIPT_DIR}/ds_config_zero2.json"

DATASETS="dataset_train%100"
DATA_FLATTEN=False

mkdir -p ${OUTPUT_DIR}
mkdir -p ${PROJECT_DIR}/logs

# 训练配置:
#   tune_mm_vision=False (冻结视觉编码器)
#   tune_mm_mlp=True  (训练 MLP merger)
#   tune_mm_llm=True  (训练 LLM，配合 LoRA 控制显存)
#   use_lora=True     (LoRA 微调 LLM，显存从 80GB+ 降至 ~40GB)
# 使用 DeepSpeed ZeRO-2 + CPU offload 优化显存

torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         qwenvl/train/train_qwen.py \
         --model_name_or_path "${MODEL_PATH}" \
         --cache_dir ${CACHE_DIR} \
         --dataset_use ${DATASETS} \
         --data_flatten ${DATA_FLATTEN} \
         --tune_mm_mlp \
         --tune_mm_llm \
         --use_lora \
         --lora_r 64 \
         --lora_alpha 128 \
         --lora_dropout 0.05 \
         --bf16 \
         --output_dir ${OUTPUT_DIR} \
         --num_train_epochs 3 \
         --per_device_train_batch_size 1 \
         --per_device_eval_batch_size 1 \
         --gradient_accumulation_steps 8 \
         --max_pixels $((512*512*3)) \
         --min_pixels $((256*256*3)) \
         --eval_strategy no \
         --save_strategy steps \
         --save_steps 500 \
         --save_total_limit 2 \
         --learning_rate 1e-5 \
         --mm_projector_lr 2e-5 \
         --vision_tower_lr 1e-6 \
         --weight_decay 0.01 \
         --warmup_ratio 0.03 \
         --max_grad_norm 1 \
         --lr_scheduler_type cosine \
         --logging_steps 10 \
         --model_max_length 4096 \
         --gradient_checkpointing True \
         --dataloader_num_workers 4 \
         --run_name qwen3vl_lora_finetune \
         --report_to none \
         --optim adamw_torch \
         --deepspeed ${DS_CONFIG} \
         2>&1 | tee ${LOG_FILE}