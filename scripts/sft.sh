#!/bin/bash

# ══════════════════════════════════════════════════════════════════════
# Qwen3-VL SFT 训练脚本
# 任务：1张卫星图 + 5张侧视图 → 结构化建筑描述（3-5句）
# 显存：80GB VRAM，内存：64GB（workers=0）/ 96GB+（workers=2）
# 用法：cd /opt/data/private/qwen3-vl-master/qwen3-vl && bash scripts/sft.sh
# ══════════════════════════════════════════════════════════════════════

# ── 项目根目录（所有相对路径的基准）────────────────────────────────
PROJECT_ROOT="/opt/data/private/qwen3-vl-master/qwen3-vl"

# ── 切换到项目根目录，保证相对路径正确 ──────────────────────────────
cd "${PROJECT_ROOT}" || { echo "[ERROR] 项目目录不存在: ${PROJECT_ROOT}"; exit 1; }

# ── 路径配置 ─────────────────────────────────────────────────────────
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}

MODEL_PATH="${PROJECT_ROOT}/pretrained/Qwen3-VL-8B-Instruct"
CACHE_DIR="${PROJECT_ROOT}/cache"
DATASETS=${DATASETS:-"train_dataset"}
DATA_FLATTEN=${DATA_FLATTEN:-False}
OUTPUT_DIR="${PROJECT_ROOT}/output/qwen3vl_lora"
DS_CONFIG="${PROJECT_ROOT}/scripts/ds_config_zero2.json"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "[INFO] PROJECT_ROOT : ${PROJECT_ROOT}"
echo "[INFO] MODEL_PATH   : ${MODEL_PATH}"
echo "[INFO] OUTPUT_DIR   : ${OUTPUT_DIR}"
echo "[INFO] DS_CONFIG    : ${DS_CONFIG}"
echo "[INFO] LOG_FILE     : ${LOG_FILE}"
echo "[INFO] PWD          : $(pwd)"

torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         "${PROJECT_ROOT}/qwenvl/train/train_qwen.py" \
         --model_name_or_path "${MODEL_PATH}" \
         --cache_dir "${CACHE_DIR}" \
         --dataset_use "${DATASETS}" \
         --data_flatten ${DATA_FLATTEN} \
         \
         `# ── 训练模块控制 ──────────────────────────────────────────` \
         --tune_mm_mlp \
         --tune_mm_llm \
         \
         `# ── LoRA 配置 ─────────────────────────────────────────────` \
         `# r=64 对结构固定的短文本任务已足够，比 r=128 省约25%可训练参数` \
         --use_lora \
         --lora_r 128 \
         --lora_alpha 256 \
         --lora_dropout 0.1 \
         \
         `# ── 精度 ──────────────────────────────────────────────────` \
         --bf16 \
         --output_dir "${OUTPUT_DIR}" \
         \
         `# ── Batch & 梯度 ──────────────────────────────────────────` \
         `# 有效 batch = per_device(2) × grad_accum(8) = 16` \
         --per_device_train_batch_size 2 \
         --per_device_eval_batch_size 1 \
         --gradient_accumulation_steps 8 \
         \
         `# ── 序列长度 ──────────────────────────────────────────────` \
         `# 6张512×512图 → ~1944 vision token + prompt/response ~300` \
         `# 原值 2048 会截断输入，3072 留有安全余量` \
         --model_max_length 3072 \
         \
         `# ── 图像分辨率范围 ────────────────────────────────────────` \
         `# processor 使用 H×W（不含通道数）` \
         --max_pixels $((512*512)) \
         --min_pixels $((256*256)) \
         \
         `# ── 训练轮数 ──────────────────────────────────────────────` \
         --num_train_epochs 8 \
         \
         `# ── 验证策略 ──────────────────────────────────────────────` \
         --eval_strategy steps \
         --eval_steps 100 \
         \
         `# ── 保存策略 ──────────────────────────────────────────────` \
         --save_strategy steps \
         --save_steps 200 \
         --save_total_limit 3 \
         --load_best_model_at_end True \
         --metric_for_best_model eval_loss \
         \
         `# ── 学习率 ────────────────────────────────────────────────` \
         --learning_rate 5e-6 \
         --mm_projector_lr 1e-5 \
         \
         `# ── 正则化 ────────────────────────────────────────────────` \
         --weight_decay 0.01 \
         --max_grad_norm 1 \
         \
         `# ── Warmup ────────────────────────────────────────────────` \
         --warmup_steps 100 \
         \
         `# ── LR 调度 ───────────────────────────────────────────────` \
         --lr_scheduler_type cosine \
         \
         `# ── 日志 ──────────────────────────────────────────────────` \
         --logging_steps 10 \
         --run_name qwen3vl_lora_finetune \
         --report_to none \
         \
         `# ── 其他 ──────────────────────────────────────────────────` \
         --gradient_checkpointing True \
         --optim adamw_torch \
         --dataloader_num_workers 2 \
         --deepspeed "${DS_CONFIG}" \
         2>&1 | tee "${LOG_FILE}"