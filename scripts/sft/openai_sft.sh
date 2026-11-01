#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/scripts/common/load_db_config.sh"

# API配置现在�?assets/model.toml 读取
# 不再需要在这里设置环境变量

# 用法�?
#   1. 训练微调（默认模式）
#        ./oai_sft.sh
#   2. 处理已有 job（查�?取消等）
#        ./oai_sft.sh <action> [job_id] [model_name] [events_output.md]
#        action �?{status, events, checkpoints, cancel, all}
#        job_id 默认使用脚本中的 JOB_ID 变量
#        model_name 用于读取 assets/model.toml 中的 API 凭据，默认使�?MODEL 变量

#        events_output.md 可选，仅对 events/all 有效，将完整事件写入该 md 文件（不截断）

# 定义变量
PROMPT="social_science_rqcontext"
SUBJECT="ECONOMICS ob"
TYPES="study"
YEARS=""  # 可选：如 "2021 2022"
MODEL="gpt-4.1-nano-2025-04-14"
JOB_ID="4.1-nano-social_science_rqcontext"

# 数据平衡（可选）：每类最�?N 条；为空则不启用
BALANCE=""
BALANCE_SEED=42
BALANCE_STRATEGY="random"  # random|year_desc：year_desc 按年份降序优先

# 输出/命名后缀，如 old / new，会拼到 suffix 里
SUFFIX=""

# 多学科缩写：每科取前 3 字母大写，不足 3 则整词大写，用下划线连接 → ECO_SOC_OB
SUBJECT_PREFIX=""
for word in $SUBJECT; do
  if [ ${#word} -lt 3 ]; then
    SUBJECT_PREFIX="${SUBJECT_PREFIX:+${SUBJECT_PREFIX}_}$(echo "$word" | tr '[:lower:]' '[:upper:]')"
  else
    SUBJECT_PREFIX="${SUBJECT_PREFIX:+${SUBJECT_PREFIX}_}$(echo "${word:0:3}" | tr '[:lower:]' '[:upper:]')"
  fi
done

if [ $# -ge 1 ]; then
  action="$1"
  job_id="${2:-$JOB_ID}"
  case "$action" in
    status|events|checkpoints|cancel|all)
      model_name="${3:-$MODEL}"
      events_md="${4:-}"
      if [ -n "$events_md" ]; then
        python -m src.core.sft.openai.jobs "$action" "$job_id" "$model_name" "$events_md"
      else
        python -m src.core.sft.openai.jobs "$action" "$job_id" "$model_name"
      fi
      exit
      ;;
  esac
fi

DB_NAME="RQ"
load_db_config "${REPO_ROOT}" "${DB_NAME}" || exit 1
CONNECTION_STRING="${MONGODB_CONNECTION_STRING}"

# 动态拼接可选参�?
BALANCE_ARGS=()
if [ -n "${BALANCE}" ]; then
  BALANCE_ARGS+=(--balance "${BALANCE}" --balance_seed "${BALANCE_SEED:-42}" --balance_strategy "${BALANCE_STRATEGY:-random}")
fi

# 年份可选参数
YEARS_ARGS=()
if [ -n "${YEARS}" ]; then
  YEARS_ARGS+=(--years ${YEARS})
fi

python -m src.practices.article.sft \
  --trainer openai \
  --model "${MODEL}" \
  --db_name RItest \
  --connection_string "${CONNECTION_STRING}" \
  --split train \
  --prompt ${PROMPT} \
  --subjects ${SUBJECT} \
  --types ${TYPES} \
  "${YEARS_ARGS[@]}" \
  --entry rq_with_context \
  --batch_size 32 \
  --epochs 3 \
  --lr_mult 1 \
  --n_checkpoints 3 \
  --suffix "${SUBJECT_PREFIX}_${PROMPT}-${SUFFIX}" \
  --output_dir ./finetune/${SUBJECT_PREFIX}_${PROMPT} \
  "${BALANCE_ARGS[@]}"
