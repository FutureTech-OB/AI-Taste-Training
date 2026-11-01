#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/scripts/common/load_db_config.sh"

# Pairwise (dual) validation script.
# 4 tiers → 6 pair types (两两配对); samples 50 pairs per type (300 total).
# Model picks the article with higher publication potential; correct = higher tier.

# ---- Model list ----
# $1: model(s), space-separated  e.g. ./dual_val.sh "model_a model_b"
MODELS="google/gemini-3.1-pro-preview"
[ -n "$1" ] && MODELS="$1"

PROVIDER="openai.openrouter"
TEMPERATURE="${2:-1.0}"        # $2: temperature (default 0)
MAX_CONCURRENT="${3:-20}"      # $3: max concurrent requests
THINKING_MODEL="${4:-True}"   # $4: is the model a thinking model (True/False)
ENABLE_THINKING="${5:-True}"  # $5: enable thinking output  (True/False)

# ---- Pairwise sampling: 6 tier-pair types, 50 per type, fixed seed ----
SEED=32
NUM_PER_PAIR_TYPE=50

# ---- Data ----
DATA_SOURCE="jsonl"  # jsonl 或 mongodb
JSONL_PATH="C:\\Users\\45391\\codes\\RQ\\reports\\data\\model_predictions\\new_sft\\120_gpt.jsonl"
DB_NAME="RQ0"
load_db_config "${REPO_ROOT}" "${DB_NAME}" || exit 1
CONNECTION_STRING="${MONGODB_CONNECTION_STRING}"
SPLIT="validate"
SUBJECT="ob"
TYPES="study"

# ---- Task config ----
PROMPT="ob_rqcontext_dual"
ENTRY="rq_with_context"

OUT_DIR="./outcome/dual_${PROMPT}_${SUBJECT}"

for MODEL in ${MODELS}; do
  echo "========== Dual Validation: ${MODEL} =========="
  python -m src.practices.article.dual_validation \
    --model "${MODEL}" \
    --provider "${PROVIDER}" \
    --thinking_model "${THINKING_MODEL}" \
    --enable_thinking "${ENABLE_THINKING}" \
    --config_path ./assets/model.toml \
    --prompt "${PROMPT}" \
    --entry "${ENTRY}" \
    --subjects "${SUBJECT}" \
    ${TYPES:+--types ${TYPES}} \
    --data_source "${DATA_SOURCE}" \
    --jsonl_path "${JSONL_PATH}" \
    --split "${SPLIT}" \
    --db_name "${DB_NAME}" \
    --connection_string "${CONNECTION_STRING}" \
    --seed "${SEED}" \
    --num_per_pair_type "${NUM_PER_PAIR_TYPE}" \
    --max_concurrent "${MAX_CONCURRENT}" \
    --temperature "${TEMPERATURE}" \
    --output_dir "${OUT_DIR}"
done
