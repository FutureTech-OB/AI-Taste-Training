#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/scripts/common/load_db_config.sh"
# Article SFT 训练启动脚本
# 通过 python -m src.practices.article.sft 统一入口
# 脚本负责：数据加载（MongoDB/JSONL）→ 消息构建 → 写出 JSONL → 启动 DeepSpeed 子进程
#
# 用法（切换 TRAINER 变量即可）：
#   bash scripts/sft/article_sft.sh
#
set -e

# ── 训练器：openai | deepspeed ──────────────────────────────────────────────
TRAINER="deepspeed"

# ── 数据源：mongodb | jsonl ─────────────────────────────────────────────────
DATA_SOURCE="jsonl"

# ── MongoDB 配置（DATA_SOURCE=mongodb 时使用） ──────────────────────────────
DB_NAME="RItime"
load_db_config "${REPO_ROOT}" "${DB_NAME}" || exit 1
CONNECTION_STRING="${MONGODB_CONNECTION_STRING}"

# ── JSONL 配置（DATA_SOURCE=jsonl 时使用） ──────────────────────────────────
JSONL_PATH="./data/ob_recent_train.jsonl"
# 验证集：留空则用 MongoDB split=validate；指定则用该 JSONL 作 eval（与 DATA_SOURCE 无关）
EVAL_JSONL_PATH="./data/ob_120_bench.jsonl"

# ── 数据过滤 ────────────────────────────────────────────────────────────────
SPLIT="train"
SUBJECTS="ob"
TYPES="study"       # 留空则不过滤：TYPES=""
YEARS=""            # 可选：如 "2024 2025"，留空则不过滤年份
PROMPT="ob_rqcontext"
ENTRY="rq_with_context"
TARGET_FIELD="rank"

# ── 公共训练超参 ────────────────────────────────────────────────────────────
# 全局 batch = per_device × 卡数 × grad_accum；当前为 7 卡，用 28 使 grad_accum=1；8 卡时改为 32
BATCH_SIZE=32
EPOCHS=2
OUTPUT_DIR="./finetune/${PROMPT}_${SUBJECTS}_30B_${LR}"
# 按类别平衡：每类最多 N 条（不足则全取），留空则不 balance
BALANCE="600"
BALANCE_SEED=42
BALANCE_STRATEGY="random"   # random|year_desc：year_desc 按年份降序优先

# ── OpenAI 专有配置 ─────────────────────────────────────────────────────────
OAI_MODEL="gpt-4.1-nano-2025-04-14"
OAI_SUFFIX="${SUBJECTS}-${PROMPT}"
OAI_LR_MULT=1.5
OAI_N_CHECKPOINTS=3

# ── DeepSpeed 专有配置 ──────────────────────────────────────────────────────
# 只用 7 卡时排除 GPU 1（例如留一张做推理），用下面这行；用满 8 卡则注释掉，并把上面 BATCH_SIZE 改为 32
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MODEL_PATH="/workspace/gongziqin/228/RQ/finetune/ob_rqcontext_ob_30B_/checkpoint-226"
PER_DEVICE_BATCH_SIZE=4   # 7 卡时有效 batch=BATCH_SIZE=28（4×7×1）；8 卡时若 BATCH_SIZE=32 则 4×8×1=32
LR=1e-5
MAX_LENGTH=2560
DS_CONFIG="/workspace/gongziqin/228/RQ/assets/ds_config_zero3.json"
MASTER_PORT=28500
SAVE_STRATEGY="epoch"
SAVE_STEPS=100
# 仅保存模型权重，不保存优化器/调度器/rng（TRL save_only_model，省空间；无法断点续训）
SAVE_ONLY_MODEL=""   # 设为非空则开启，例如: SAVE_ONLY_MODEL="1"

# ── 环境 ────────────────────────────────────────────────────────────────────
export PYTHONPATH="/workspace/gongziqin/228/RQ:${PYTHONPATH}"
mkdir -p "${OUTPUT_DIR}"

# ── 公共参数 ─────────────────────────────────────────────────────────────────
COMMON_ARGS=(
    --trainer        "${TRAINER}"
    --data_source    "${DATA_SOURCE}"
    --split          "${SPLIT}"
    --subjects       ${SUBJECTS}
    --prompt         "${PROMPT}"
    --entry          "${ENTRY}"
    --target_field   "${TARGET_FIELD}"
    --batch_size     ${BATCH_SIZE}
    --epochs         ${EPOCHS}
    --output_dir     "${OUTPUT_DIR}"
)

if [ -n "${TYPES}" ]; then
    COMMON_ARGS+=(--types ${TYPES})
fi
if [ -n "${YEARS}" ]; then
    COMMON_ARGS+=(--years ${YEARS})
fi
if [ -n "${BALANCE}" ]; then
    COMMON_ARGS+=(--balance "${BALANCE}" --balance_seed "${BALANCE_SEED:-42}" --balance_strategy "${BALANCE_STRATEGY:-random}")
fi

if [ "${DATA_SOURCE}" = "mongodb" ]; then
    COMMON_ARGS+=(
        --db_name           "${DB_NAME}"
        --connection_string "${CONNECTION_STRING}"
    )
else
    COMMON_ARGS+=(
        --jsonl_path "${JSONL_PATH}"
    )
fi
if [ -n "${EVAL_JSONL_PATH}" ]; then
    COMMON_ARGS+=(--eval_jsonl_path "${EVAL_JSONL_PATH}")
fi

# ── 训练器特有参数 ───────────────────────────────────────────────────────────
if [ "${TRAINER}" = "openai" ]; then
    python -m src.practices.article.sft \
        "${COMMON_ARGS[@]}" \
        --model          "${OAI_MODEL}" \
        --suffix         "${OAI_SUFFIX}" \
        --lr_mult        ${OAI_LR_MULT} \
        --n_checkpoints  ${OAI_N_CHECKPOINTS}
else
    python -m src.practices.article.sft \
        "${COMMON_ARGS[@]}" \
        --model_path             "${MODEL_PATH}" \
        --attn_implementation    "flash_attention_2" \
        --save_strategy          "${SAVE_STRATEGY}" \
        --save_steps             "${SAVE_STEPS}" \
        --per_device_batch_size  ${PER_DEVICE_BATCH_SIZE} \
        --lr                     ${LR} \
        --max_length             ${MAX_LENGTH} \
        --deepspeed_config       "${DS_CONFIG}" \
        --gradient_checkpointing

    # LoRA 示例（去掉注释即可）：
    #   --use_peft
    #   --lora_r 16
    #   --lora_alpha 32
fi
