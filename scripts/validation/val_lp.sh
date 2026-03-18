#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/scripts/common/load_db_config.sh"

# API 配置现在从 assets/model.toml 读取

PROMPT="social_science_rqcontext"
SUBJECTS=(ECONOMICS)
SUBJECT_TAG=$(IFS=_; echo "${SUBJECTS[*]}")

# 支持多个模型：空格分隔
# 若传第 1 个参数则覆盖此处列表
MODELS="ft:gpt-4.1-nano-2025-04-14:personal:eco-ob-social-scie:DJuAjWUp"
[ -n "$1" ] && MODELS="$1"

PROVIDER="openai.official"
TEMPERATURE="${2:-0.0}"
MAX_CONCURRENT="${3:-20}"

DATA_SOURCE="mongodb"
JSONL_PATH="./reports/data/model_predictions/new_sft/120_gpt_crq.jsonl"
DB_NAME="RItest"
TYPES=""
load_db_config "${REPO_ROOT}" "${DB_NAME}" || exit 1
CONNECTION_STRING="${MONGODB_CONNECTION_STRING}"

OUT_DIR="./outcome/${PROMPT}_${SUBJECT_TAG}"
DISTILL_BASE="./distill/${PROMPT}_${SUBJECT_TAG}"

for MODEL in ${MODELS}; do
  echo "========== Validating model: ${MODEL} =========="
  python -m src.practices.article.validation \
    --model "${MODEL}" \
    --provider "${PROVIDER}" \
    --thinking_model False \
    --enable_thinking False \
    --enable_logp True \
    --max_concurrent "${MAX_CONCURRENT}" \
    --split validate \
    --prompt "${PROMPT}" \
    --subjects "${SUBJECTS[@]}" \
    ${TYPES:+--types ${TYPES}} \
    --entry rq_with_context \
    --data_source "${DATA_SOURCE}" \
    --jsonl_path "${JSONL_PATH}" \
    --db_name "${DB_NAME}" \
    --connection_string "${CONNECTION_STRING}" \
    --config_path ./assets/model.toml \
    --temperature "${TEMPERATURE}" \
    --output_dir "${OUT_DIR}" \
    ${DISTILL_BASE:+--distill_dir "${DISTILL_BASE}"} \
    # --overwrite_cache
done
