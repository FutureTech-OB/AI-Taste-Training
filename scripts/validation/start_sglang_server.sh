#!/bin/bash
# 启动SGLang服务器加载checkpoint

set -e

# 配置
CHECKPOINT_PATH="${1:-/workspace/gongziqin/228/RQ/OB_models/ob_rqcontext_ob_30B/checkpoint-484}"
PORT="${2:-8000}"
HOST="${3:-0.0.0.0}"

# 检查checkpoint是否存在
if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "错误: checkpoint路径不存在: ${CHECKPOINT_PATH}"
    exit 1
fi

echo "启动SGLang服务器..."
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "端口: ${PORT}"
echo "主机: ${HOST}"

# 启动sglang服务器
python -m sglang.launch_server \
    --model-path "${CHECKPOINT_PATH}" \
    --port "${PORT}" \
    --host "${HOST}" \
    --trust-remote-code \
    --dtype bfloat16

