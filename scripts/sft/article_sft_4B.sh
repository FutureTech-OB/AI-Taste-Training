#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/scripts/common/load_db_config.sh"
# Article SFT 训练启动脚本 —— Qwen3-4B
# 8× A800-80GB，ZeRO-2，全参数微调
#
# 显存估算：
#   模型参数  ~4B × 2B (bf16) = 8 GB
#   ZeRO-2 优化器状态分片后每卡 ~8 GB
#   激活值（per_device_batch=32, len=1024）~8 GB
#   合计每卡 ~24 GB，80 GB 显存充裕
#
set -e

# ── 训练器：openai | deepspeed ──────────────────────────────────────────────
TRAINER="deepspeed"

# ── 数据源：mongodb | jsonl ─────────────────────────────────────────────────
DATA_SOURCE="jsonl"

# ── MongoDB 配置（DATA_SOURCE=mongodb 时使用） ──────────────────────────────
DB_NAME="RIOB"
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
PROMPT="ob_rqcontext"
ENTRY="rq_with_context"
TARGET_FIELD="rank"

# ── 公共训练超参 ────────────────────────────────────────────────────────────
# 全局 batch = per_device × 卡数 × grad_accum；8 卡用 32（4×8×1）；若用 7 卡则设 BATCH_SIZE=28
# 4B 模型收敛快，3 epoch 通常足够，可视 loss 曲线调整
BATCH_SIZE=28
EPOCHS=3
OUTPUT_DIR="./finetune/${PROMPT}_${SUBJECTS}_4B_${LR}"
# 按类别平衡：每类最多 N 条（不足则全取），留空则不 balance
BALANCE=""
BALANCE_SEED=42
BALANCE_STRATEGY="year_desc"

# ── DeepSpeed 专有配置 ──────────────────────────────────────────────────────
# 只用 7 卡时排除 GPU 1，取消下面注释并设上面 BATCH_SIZE=28；用满 8 卡则保持注释
export CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7
MODEL_PATH="/workspace/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507"
PER_DEVICE_BATCH_SIZE=4   # 8 卡时有效 batch=32（4×8×1）；7 卡时 BATCH_SIZE=28
LR=1e-4                    # 4B 参数量少，可用略大学习率
MAX_LENGTH=2560             # RQ 文本通常 <1024 token，无需 2048
DS_CONFIG="/workspace/gongziqin/228/RQ/assets/ds_config_zero2.json"
MASTER_PORT=28501           # 与 30B 脚本错开端口
# 存盘：epoch=每 epoch 结束存一次；steps=按 save_steps 步数存
SAVE_STRATEGY="epoch"
SAVE_STEPS=100              # save_strategy=steps 时生效

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
if [ -n "${BALANCE}" ]; then
    COMMON_ARGS+=(--balance "${BALANCE}" --balance_seed "${BALANCE_SEED:-42}")
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

# ── 启动训练 ─────────────────────────────────────────────────────────────────
python -m src.practices.article.sft \
    "${COMMON_ARGS[@]}" \
    --model_path             "${MODEL_PATH}" \
    --attn_implementation   sdpa \
    --save_strategy          "${SAVE_STRATEGY}" \
    --save_steps             "${SAVE_STEPS}" \
    --per_device_batch_size  ${PER_DEVICE_BATCH_SIZE} \
    --lr                     ${LR} \
    --max_length             ${MAX_LENGTH} \
    --deepspeed_config       "${DS_CONFIG}" \
    --master_port            ${MASTER_PORT} \
    --gradient_checkpointing

# LoRA 示例（参数量小时全参数微调更优，但若显存仍紧张可启用）：
#   --use_peft
#   --lora_r 16
#   --lora_alpha 32
