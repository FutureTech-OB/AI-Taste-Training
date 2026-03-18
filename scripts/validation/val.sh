#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/scripts/common/load_db_config.sh"

# API配置现在从 assets/model.toml 读取
# 不再需要在这里设置环境变量

# 定义变量
PROMPT="social_science_rqcontext"
SUBJECT="SOCIOLOGY"
# 支持多个模型：空格分隔，如 MODELS="moonshotai/kimi-k2.5 openai/gpt-4o"
# 若传第 1 个参数则覆盖此处列表，如 ./val.sh "model1 model2" 或 ./val.sh "model1"
MODELS="google/gemini-3.1-pro-preview"
[ -n "$1" ] && MODELS="$1"
PROVIDER="openai.openrouter"  # Provider 配置（硬编码）
TEMPERATURE="${2:-1.0}"  # 可选：第2个参数传 temperature
MAX_CONCURRENT="${3:-15}"  # 可选：第3个参数传最大并发（默认 1，避免 429）

# 数据源配置
DATA_SOURCE="mongodb"  # 或 "mongodb"
JSONL_PATH="C:\\Users\\45391\\codes\\RQ\\reports\\data\\model_predictions\\120_frontier.jsonl"  # Windows 路径格式
DB_NAME="RItest"  # MongoDB 模式时使用
TYPES=""  # 文章类型过滤（留空则不过滤，多个类型空格分隔如 "study review"）
# MongoDB 连接字符串（可选，如未提供则使用默认连接）
# 格式: mongodb://username:password@host:port/database?authSource=admin
load_db_config "${REPO_ROOT}" "${DB_NAME}" || exit 1
CONNECTION_STRING="${MONGODB_CONNECTION_STRING}"

OUT_DIR="./outcome/${PROMPT}_${SUBJECT}"
DISTILL_DIR="./distill/${PROMPT}_${SUBJECT}_${MODEL}"  # 蒸馏数据目录（留空则不保存）

# 对每个模型依次运行验证
for MODEL in ${MODELS}; do
  echo "========== Validating model: ${MODEL} =========="
  python -m src.practices.article.validation \
    --model "${MODEL}" \
    --provider "${PROVIDER}" \
    --thinking_model True \
    --enable_thinking True \
    --enable_logp False \
    --avg_n 4 \
    --max_concurrent "${MAX_CONCURRENT}" \
    --split validate \
    --prompt "${PROMPT}" \
    --subjects "${SUBJECT}" \
    ${TYPES:+--types ${TYPES}} \
    --entry rq_with_context \
    --data_source "${DATA_SOURCE}" \
    --jsonl_path "${JSONL_PATH}" \
    --db_name "${DB_NAME}" \
    --connection_string "${CONNECTION_STRING}" \
    --config_path ./assets/model.toml \
    --temperature "${TEMPERATURE}" \
    --max_tokens 8192 \
    --output_dir "${OUT_DIR}" \
    ${DISTILL_DIR:+--distill_dir "${DISTILL_DIR}"} \
    # --overwrite_cache
done
