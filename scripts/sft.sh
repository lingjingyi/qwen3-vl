#!/bin/bash
# ══════════════════════════════════════════════════════════════════════
# Qwen3-VL SFT 训练脚本
# 用法：bash scripts/sft.sh
#
# 显存/内存需求（基于当前配置）：
#   VRAM：~30GB（80GB GPU 使用率约 37%，含 SemanticSlot 约 31GB）
#   RAM ：dataloader_num_workers=0 → 64GB+
#         dataloader_num_workers=2 → 96GB+
# ══════════════════════════════════════════════════════════════════════


# ┌─────────────────────────────────────────────────────────────────────
# │  ★ 用户配置区 — 所有训练参数在此修改
# └─────────────────────────────────────────────────────────────────────

# ── [实验标识] ──────────────────────────────────────────────────────────
# 消融示例：
#   "rsgpt_baseline"      USE_STYLE_PREFIX=False USE_SEMANTIC_SLOT=False TUNE_MM_VISION=False
#   "rsgpt_slot_only"     USE_STYLE_PREFIX=False USE_SEMANTIC_SLOT=True
#   "rsgpt_style_only"    USE_STYLE_PREFIX=True  USE_SEMANTIC_SLOT=False
#   "rsgpt_full"          USE_STYLE_PREFIX=True  USE_SEMANTIC_SLOT=True
#   "rsgpt_full_vit"      USE_STYLE_PREFIX=True  USE_SEMANTIC_SLOT=True TUNE_MM_VISION=True
#   "rsgpt_no_count"      USE_COUNT_HEAD=False  （对比有无计数模块）
#   "rsgpt_no_slot_attn"  STYLE_USE_SLOT_ATTENTION=False  （对比有无位置感知）
EXP_NAME="rsgpt_count"
 
# ── [续训控制] ──────────────────────────────────────────────────────────
# False（默认）：每次 bash sft.sh 都从头全新训练
#   若 output_dir 已有旧 checkpoint，自动备份后再训练，无需手动删除
# True：从 output_dir 中最新有效 checkpoint 恢复续训
#   若无有效 checkpoint，自动从头开始
RESUME_FROM_CHECKPOINT=False

# ── [路径] ─────────────────────────────────────────────────────────────
CONDA_ENV="qwenvl_ft"
PROJECT_ROOT="/opt/data/private/qwen3-vl-master/qwen3-vl"
MODEL_PATH="${PROJECT_ROOT}/pretrained/Qwen3-VL-8B-Instruct"
CACHE_DIR="${PROJECT_ROOT}/cache"
OUTPUT_DIR="${PROJECT_ROOT}/output/${EXP_NAME}"
DS_CONFIG="${PROJECT_ROOT}/scripts/ds_config_zero2.json"
LOG_DIR="${PROJECT_ROOT}/logs"

# ── [分布式] ────────────────────────────────────────────────────────────
NPROC_PER_NODE=1
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29500

# ── [数据] ──────────────────────────────────────────────────────────────
DATASETS="rsgpt_train_aligned" #train_dataset
DATA_FLATTEN=False
MAX_PIXELS=$((512 * 512))    # 262144 → 16×16 merged → 256 个视觉 token
MIN_PIXELS=$((256 * 256))    # 65536

# ── [视觉模块] ──────────────────────────────────────────────────────────
# ★ 解冻 ViT 后几层，改善颜色/纹理/属性描述
# True 约新增 +5GB 显存（含 optimizer），建议数据量 > 2000 条时开启
TUNE_MM_VISION=False
TUNE_MM_MLP=True
TUNE_MM_LLM=True

# ── [LoRA] ──────────────────────────────────────────────────────────────
USE_LORA=True
LORA_R=256
LORA_ALPHA=512
LORA_DROPOUT=0.15

# ── [SemanticSlotToken 模块] ────────────────────────────────────────────
#   消融实验示例：
#     USE_SEMANTIC_SLOT=False  → baseline（无 slot）
#     SLOT_NUM_SLOTS=4         → K=4 消融
#     USE_POSITION_INFO=True   → 开启空间位置编码消融
USE_SEMANTIC_SLOT=True
SLOT_NUM_SLOTS=9         # slot 数量 K（消融：4 / 8 / 12）★ 9 = 3×3 网格，精确对应 9 个空间区域
SLOT_NUM_ITERATIONS=2    # 迭代轮数（消融：1 / 2 / 3）
SLOT_LR=3e-6             # slot 独立学习率（通常高于 LLM LR）
USE_POSITION_INFO=True    # ★ 开启空间位置偏置，改善方位描述
USE_COUNT_HEAD=True       # ★ 开启无监督计数感知模块

# ── [StylePrefixToken] ──────────────────────────────────────────────────
# 消融：USE_STYLE_PREFIX=False → 禁用 StylePrefix（其余不变）
USE_STYLE_PREFIX=True
STYLE_NUM_TOKENS=8      # K 值（消融：2 / 4 / 6 / 8）
# style_lr 建议比主 LR 大 100-1000 倍（style token 参数量极小 K×D）
STYLE_LR=1e-4
STYLE_USE_SLOT_ATTENTION=True   # ★ style token 感知 slot 空间信息
 
# ★ 更新初始化句子（覆盖 RSGPT 典型句式）
# 每条句子对应一个 style token 的初始语义，越典型越好
STYLE_INIT_SENTENCES='This is a high-resolution aerial image showing a residential area.;This is a panchromatic satellite image showing a large area of farmland.;This is a high-resolution remote sensing image showing a forest.;This is an aerial image showing a harbor with several boats docked.;In the center of the image, there is a road running east to west with several cars.;In the upper left corner of the image, there is a parking lot with numerous vehicles.;Below the road is a large grassy area with several trees scattered throughout.;On the right side of the image, there are two gray-roofed buildings surrounded by trees.;Some vehicles are parked next to the houses and along the roads.;There are a total of seven buses parked neatly in a row.;The bottom-right corner of the image shows a body of water with ships.;Several small airplanes can be seen parked on the tarmac.'

# ── [批次与序列长度] ────────────────────────────────────────────────────
PER_DEVICE_TRAIN_BATCH=2
PER_DEVICE_EVAL_BATCH=1
GRAD_ACCUM_STEPS=8       # 等效 batch = 2 × 8 = 16
# 256(visual) + 9(slot≈0) + 12(style) + ~100(text) + margin = 3120
MODEL_MAX_LENGTH=3250

# ── [训练轮次与评估] ────────────────────────────────────────────────────
NUM_EPOCHS=5
# ★ val_ratio=0, 关闭所有 eval 相关
EVAL_STRATEGY=no
SAVE_STRATEGY=steps           # ★ 补上这一行 (原缺失)
SAVE_STEPS=180                # ★ 约每 epoch 存一次
SAVE_TOTAL_LIMIT=2            # ★ 只保留 2 个 checkpoint, 省磁盘
LOAD_BEST_MODEL=False         # ★ 无 eval 就不能 load best

# ── [学习率] ────────────────────────────────────────────────────────────
LEARNING_RATE=5e-6
MM_PROJECTOR_LR=1e-5     # Merger MLP 独立学习率
# ★ ViT 学习率（比主 LR 低一个数量级，避免 ViT 过拟合）
VISION_TOWER_LR=1e-6
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1
WARMUP_STEPS=200
LR_SCHEDULER=cosine

# ── [日志与优化器] ──────────────────────────────────────────────────────
LOGGING_STEPS=10
OPTIM=adamw_torch
RUN_NAME=qwen3vl_plan_d_v1
NUM_WORKERS=2            # RAM 不足时改为 0（节省约 20-30GB RAM）

# ─────────────────────────────────────────────────────────────────────
#  以下无需修改
# ─────────────────────────────────────────────────────────────────────

CONDA_ENV_PATH="/opt/conda/envs/${CONDA_ENV}"
TORCHRUN="${CONDA_ENV_PATH}/bin/torchrun"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

# ── 环境验证 ────────────────────────────────────────────────────────────
if [ ! -f "${TORCHRUN}" ]; then
    echo "[ERROR] torchrun not found: ${TORCHRUN}"
    echo "[ERROR] 请确认 conda 环境名: ${CONDA_ENV}"
    exit 1
fi

cd "${PROJECT_ROOT}" || { echo "[ERROR] 项目目录不存在: ${PROJECT_ROOT}"; exit 1; }

for path in "${MODEL_PATH}" "${DS_CONFIG}"; do
    if [ ! -e "${path}" ]; then
        echo "[ERROR] 路径不存在: ${path}"
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${CACHE_DIR}"

# ── 启动信息 ────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════"
echo "  PROJECT_ROOT : ${PROJECT_ROOT}"
echo "  MODEL_PATH   : ${MODEL_PATH}"
echo "  OUTPUT_DIR   : ${OUTPUT_DIR}"
echo "  LOG_FILE     : ${LOG_FILE}"
echo "  DATASETS     : ${DATASETS}"
echo "  ViT            : TUNE=${TUNE_MM_VISION}  lr=${VISION_TOWER_LR}"
echo "  SemanticSlot   : USE=${USE_SEMANTIC_SLOT}  K=${SLOT_NUM_SLOTS}  pos=${USE_POSITION_INFO}  count=${USE_COUNT_HEAD}"
echo "  StylePrefix    : USE=${USE_STYLE_PREFIX}  K=${STYLE_NUM_TOKENS}  slot_attn=${STYLE_USE_SLOT_ATTENTION}"
echo "═══════════════════════════════════════════════════"

# ── 训练启动 ────────────────────────────────────────────────────────────
${TORCHRUN} \
    --nproc_per_node ${NPROC_PER_NODE} \
    --master_addr    ${MASTER_ADDR} \
    --master_port    ${MASTER_PORT} \
    "${PROJECT_ROOT}/qwenvl/train/train_qwen.py" \
    \
    --model_name_or_path  "${MODEL_PATH}" \
    --cache_dir           "${CACHE_DIR}" \
    --dataset_use         "${DATASETS}" \
    --data_flatten        ${DATA_FLATTEN} \
    --max_pixels          ${MAX_PIXELS} \
    --min_pixels          ${MIN_PIXELS} \
    \
    --tune_mm_vision      ${TUNE_MM_VISION} \
    --tune_mm_mlp         ${TUNE_MM_MLP} \
    --tune_mm_llm         ${TUNE_MM_LLM} \
    \
    --use_lora            ${USE_LORA} \
    --lora_r              ${LORA_R} \
    --lora_alpha          ${LORA_ALPHA} \
    --lora_dropout        ${LORA_DROPOUT} \
    \
    --use_semantic_slot   ${USE_SEMANTIC_SLOT} \
    --slot_num_slots      ${SLOT_NUM_SLOTS} \
    --slot_num_iterations ${SLOT_NUM_ITERATIONS} \
    --slot_lr             ${SLOT_LR} \
    --use_position_info   ${USE_POSITION_INFO} \
    --use_count_head         ${USE_COUNT_HEAD} \
    \
    --use_style_prefix       ${USE_STYLE_PREFIX} \
    --style_num_tokens       ${STYLE_NUM_TOKENS} \
    --style_lr               ${STYLE_LR} \
    --style_use_slot_attention  ${STYLE_USE_SLOT_ATTENTION} \
    --style_init_sentences      "${STYLE_INIT_SENTENCES}" \
    \
    --resume_training     ${RESUME_FROM_CHECKPOINT} \
    \
    --bf16                True \
    --output_dir          "${OUTPUT_DIR}" \
    --model_max_length    ${MODEL_MAX_LENGTH} \
    \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH} \
    --per_device_eval_batch_size  ${PER_DEVICE_EVAL_BATCH} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    \
    --num_train_epochs        ${NUM_EPOCHS} \
    --eval_strategy           ${EVAL_STRATEGY} \
    --save_strategy           ${SAVE_STRATEGY} \
    --save_steps              ${SAVE_STEPS} \
    --save_total_limit        ${SAVE_TOTAL_LIMIT} \
    \
    --learning_rate       ${LEARNING_RATE} \
    --mm_projector_lr     ${MM_PROJECTOR_LR} \
    --vision_tower_lr     ${VISION_TOWER_LR} \
    --weight_decay        ${WEIGHT_DECAY} \
    --max_grad_norm       ${MAX_GRAD_NORM} \
    --warmup_steps        ${WARMUP_STEPS} \
    --lr_scheduler_type   ${LR_SCHEDULER} \
    \
    --logging_steps       ${LOGGING_STEPS} \
    --run_name            ${EXP_NAME} \
    --report_to           none \
    \
    --gradient_checkpointing True \
    --optim               ${OPTIM} \
    --dataloader_num_workers ${NUM_WORKERS} \
    --deepspeed           "${DS_CONFIG}" \
    2>&1 | tee "${LOG_FILE}"