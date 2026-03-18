#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/scripts/common/load_db_config.sh"
# 使用 checkpoint 运行验证脚本
# 用法:
#   ./val_ckpt.sh [MODEL_NAME] [TEMPERATURE] [MAX_CONCURRENT]
#     - MODEL_NAME: 传入完整模型名（优先使用，不传则从 /v1/models 自动读取）
#     - TEMPERATURE: 可选，默认 0.0
#     - MAX_CONCURRENT: 可选，默认 1（避免触发 429）

set -e

# ========== 配置参数（所有参数写死在这里） ==========
# Checkpoint 路径
CHECKPOINT_PATH="/workspace/gongziqin/228/RQ/OB_models/30Bep2/qwen3-30b"

# 服务器配置
PORT=8000
HOST="0.0.0.0"

# 并行参数
TP_SIZE=2              # Tensor Parallel Size
DP_SIZE=2              # Data Parallel Size
PP_SIZE=1              # Pipeline Parallel Size
EP_SIZE=2              # Expert Parallel Size (for MoE models)
export CUDA_VISIBLE_DEVICES=4,5,6,7
# 分布式参数（多节点时使用）
NNODES=1               # Number of nodes
NODE_RANK=0            # Node rank
DIST_INIT_ADDR=""      # Distributed init address (e.g., "localhost:29500")

# 定义变量
PROMPT="ob_rqcontext"
SUBJECT="ob"
BASE_MODEL="qwen3-30b-a3b"
# 运行验证时使用的 provider（与 val.sh 一致：显式传 --provider）
PROVIDER="openai.sglang"
# 采样与并发（支持通过脚本参数覆盖；默认并发=1，避免 429）
MODEL_NAME_OVERRIDE="${1:-}"
TEMPERATURE="${2:-${TEMPERATURE:-1.0}}"
MAX_CONCURRENT="${3:-${MAX_CONCURRENT:-24}}"

# thinking 与 avg_n 控制
THINKING_MODEL="False"
ENABLE_THINKING="False"
ENABLE_LOGP="True"
AVG_N=""                # 留空则由 Python 自动推断（thinking 模式默认 8，否则 1）

# 数据源配置
DATA_SOURCE="jsonl"  # 或 "jsonl"
JSONL_PATH="/workspace/gongziqin/228/RQ/assets/unknown/120_qwen.jsonl"
DB_NAME="RQ0"  # MongoDB 模式时使用
TYPES=""
# MongoDB 连接字符串（可选，如未提供则使用默认连接）
# 格式: mongodb://username:password@host:port/database?authSource=admin
load_db_config "${REPO_ROOT}" "${DB_NAME}" || exit 1
MONGODB_CONNECTION_STRING="${MONGODB_CONNECTION_STRING}"

# ========== 端口检查和清理 ==========
# 检查并清理占用8000端口的进程
kill_process_on_port() {
    local port=$1
    # 通过 ss 获取占用端口的进程 PID
    EXIST_PID=$(ss -ltnp 2>/dev/null | awk -v p=":${port}" '$4~p {split($7,a,"[,=]"); print a[2]; exit}')
    if [ -n "${EXIST_PID}" ]; then
        echo "检测到端口 ${port} 已被进程 ${EXIST_PID} 占用，尝试结束..."
        # 尝试获取进程名称
        PROCESS_NAME=$(ps -p "${EXIST_PID}" -o comm= 2>/dev/null || echo "未知进程")
        echo "进程名称: ${PROCESS_NAME}"
        # 先尝试优雅退出
        kill "${EXIST_PID}" 2>/dev/null || true
        sleep 2
        # 检查进程是否还在运行
        if kill -0 "${EXIST_PID}" 2>/dev/null; then
            echo "进程未退出，强制结束..."
            kill -9 "${EXIST_PID}" 2>/dev/null || true
            sleep 1
        fi
        # 再次检查端口是否已释放
        EXIST_PID=$(ss -ltnp 2>/dev/null | awk -v p=":${port}" '$4~p {split($7,a,"[,=]"); print a[2]; exit}')
        if [ -z "${EXIST_PID}" ]; then
            echo "✓ 端口 ${port} 已释放"
        else
            echo "警告: 端口 ${port} 仍被占用，PID: ${EXIST_PID}"
        fi
    else
        echo "端口 ${port} 未被占用"
    fi
}

# 在脚本开始时就检查并清理端口
echo "========== 检查端口 ${PORT} =========="
kill_process_on_port ${PORT}
echo ""

# 检查checkpoint是否存在
if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "错误: checkpoint路径不存在: ${CHECKPOINT_PATH}"
    exit 1
fi

# 创建assets目录（如果不存在）
mkdir -p ./assets

# 创建或更新 model.toml（新格式：providers.*）
# 如果HOST是0.0.0.0，在URL中使用localhost
API_HOST="${HOST}"
if [ "${HOST}" = "0.0.0.0" ]; then
    API_HOST="localhost"
fi

# 检查model.toml是否存在，以及是否已有该模型配置
MODEL_TOML="./assets/model.toml"
if [ -f "${MODEL_TOML}" ]; then
    # 检查是否已存在该 provider 配置段
    if grep -q "^\[providers\.openai\.sglang\]" "${MODEL_TOML}"; then
        echo "配置文件中已存在 [providers.openai.sglang] 配置，跳过添加"
    else
        # 追加配置到文件末尾
        echo "" >> "${MODEL_TOML}"
        echo "[providers.openai.sglang]" >> "${MODEL_TOML}"
        echo "api_key = \"sk-0\"" >> "${MODEL_TOML}"
        echo "base_url = \"http://${API_HOST}:${PORT}/v1\"" >> "${MODEL_TOML}"
        echo "已追加配置到 ${MODEL_TOML}"
    fi
else
    # 文件不存在，创建新文件
    cat > "${MODEL_TOML}" <<EOF
# Provider 配置（SGLang 本地 OpenAI-compat）
[providers.openai.sglang]
api_key = "sk-0"
base_url = "http://${API_HOST}:${PORT}/v1"
EOF
    echo "已创建配置文件 ${MODEL_TOML}"
fi

echo "========== 步骤 1: 启动SGLang服务器 =========="
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "端口: ${PORT}"
echo "主机: ${HOST}"
echo "Tensor Parallel Size: ${TP_SIZE}"
echo "Data Parallel Size: ${DP_SIZE}"
echo "Pipeline Parallel Size: ${PP_SIZE}"
if [ "${EP_SIZE}" != "1" ]; then
    echo "Expert Parallel Size: ${EP_SIZE}"
fi
if [ "${NNODES}" != "1" ]; then
    echo "Nodes: ${NNODES}, Node Rank: ${NODE_RANK}"
    if [ -n "${DIST_INIT_ADDR}" ]; then
        echo "Dist Init Address: ${DIST_INIT_ADDR}"
    fi
fi

# 构建 SGLang 启动命令
# 如已有残留进程占用端口，先清理（避免旧的 sglang 进程未退出）
kill_sglang_on_port() {
    # 通过 ss 获取占用端口的进程 PID
    EXIST_PID=$(ss -ltnp 2>/dev/null | awk -v p=":${PORT}" '$4~p {split($7,a,"[,=]"); print a[2]; exit}')
    if [ -n "${EXIST_PID}" ]; then
        echo "检测到端口 ${PORT} 已被进程 ${EXIST_PID} 占用，尝试结束..."
        kill "${EXIST_PID}" 2>/dev/null || true
        sleep 1
        if kill -0 "${EXIST_PID}" 2>/dev/null; then
            echo "进程未退出，强制结束..."
            kill -9 "${EXIST_PID}" 2>/dev/null || true
        fi
    fi
}
kill_sglang_on_port

SGLANG_CMD="python -m sglang.launch_server"
SGLANG_CMD="${SGLANG_CMD} --model-path \"${CHECKPOINT_PATH}\""
SGLANG_CMD="${SGLANG_CMD} --port ${PORT}"
SGLANG_CMD="${SGLANG_CMD} --host \"${HOST}\""
SGLANG_CMD="${SGLANG_CMD} --trust-remote-code"
SGLANG_CMD="${SGLANG_CMD} --dtype bfloat16"

# 添加并行参数
if [ "${TP_SIZE}" != "1" ]; then
    SGLANG_CMD="${SGLANG_CMD} --tensor-parallel-size ${TP_SIZE}"
fi

if [ "${DP_SIZE}" != "1" ]; then
    SGLANG_CMD="${SGLANG_CMD} --data-parallel-size ${DP_SIZE}"
fi

if [ "${PP_SIZE}" != "1" ]; then
    SGLANG_CMD="${SGLANG_CMD} --pipeline-parallel-size ${PP_SIZE}"
fi

if [ "${EP_SIZE}" != "1" ]; then
    SGLANG_CMD="${SGLANG_CMD} --expert-parallel-size ${EP_SIZE}"
fi

# 添加分布式参数（多节点时）
if [ "${NNODES}" != "1" ]; then
    SGLANG_CMD="${SGLANG_CMD} --nnodes ${NNODES}"
    SGLANG_CMD="${SGLANG_CMD} --node-rank ${NODE_RANK}"
    if [ -n "${DIST_INIT_ADDR}" ]; then
        SGLANG_CMD="${SGLANG_CMD} --dist-init-addr \"${DIST_INIT_ADDR}\""
    fi
fi

echo ""
echo "执行命令:"
echo "${SGLANG_CMD}"
echo ""

# 在后台启动sglang服务器
eval "${SGLANG_CMD}" > sglang_server.log 2>&1 &

SGLANG_PID=$!
echo "SGLang服务器已启动，PID: ${SGLANG_PID}"
echo "日志文件: sglang_server.log"

# 等待服务器启动
echo "等待服务器启动（这可能需要几分钟加载模型）..."
sleep 5

# 检查服务器是否运行
TIMEOUT_ITERATIONS=300  # 150次 * 2秒 = 300秒超时
i=1
while [ $i -le ${TIMEOUT_ITERATIONS} ]; do
    # 尝试检查服务器是否响应（检查/v1/models端点）
    if curl -s "http://${API_HOST}:${PORT}/v1/models" > /dev/null 2>&1; then
        echo "✓ 服务器已就绪"
        break
    fi
    # 检查进程是否还在运行
    if ! kill -0 $SGLANG_PID 2>/dev/null; then
        echo "错误: SGLang服务器进程已退出"
        echo "查看日志:"
        tail -50 sglang_server.log
        exit 1
    fi
    if [ $i -eq ${TIMEOUT_ITERATIONS} ]; then
        echo "错误: 服务器启动超时（${TIMEOUT_ITERATIONS}次检查，共 $((TIMEOUT_ITERATIONS * 2)) 秒）"
        echo "查看日志:"
        tail -50 sglang_server.log
        kill $SGLANG_PID 2>/dev/null || true
        exit 1
    fi
    echo "等待中... ($i/${TIMEOUT_ITERATIONS})"
    sleep 2
    i=$((i + 1))
done

# 设置清理函数
cleanup() {
    echo ""
    echo "========== 清理 SGLang 进程 =========="
    
    # 方法1: 清理主进程及其进程组
    if [ -n "${SGLANG_PID}" ] && kill -0 "${SGLANG_PID}" 2>/dev/null; then
        echo "停止主进程 (PID: ${SGLANG_PID}) 及其进程组..."
        # 获取进程组ID
        PGID=$(ps -o pgid= -p "${SGLANG_PID}" 2>/dev/null | tr -d ' ')
        if [ -n "${PGID}" ]; then
            # 向整个进程组发送TERM信号
            kill -TERM -"${PGID}" 2>/dev/null || true
            sleep 2
            # 如果还有进程，强制结束
            if kill -0 "${SGLANG_PID}" 2>/dev/null; then
                kill -9 -"${PGID}" 2>/dev/null || true
            fi
        else
            # 如果无法获取进程组，直接kill主进程
            kill "${SGLANG_PID}" 2>/dev/null || true
            sleep 1
            if kill -0 "${SGLANG_PID}" 2>/dev/null; then
                kill -9 "${SGLANG_PID}" 2>/dev/null || true
            fi
        fi
        wait "${SGLANG_PID}" 2>/dev/null || true
    fi
    
    # 方法2: 查找并清理所有 sglang 相关进程（包括子进程）
    echo "查找所有 sglang 相关进程..."
    SGLANG_PIDS=$(pgrep -f "sglang.launch_server" 2>/dev/null || true)
    if [ -n "${SGLANG_PIDS}" ]; then
        echo "发现 sglang 进程: ${SGLANG_PIDS}"
        for pid in ${SGLANG_PIDS}; do
            # 检查进程是否还在运行
            if kill -0 "${pid}" 2>/dev/null; then
                echo "  清理进程 ${pid}..."
                kill "${pid}" 2>/dev/null || true
            fi
        done
        sleep 2
        # 强制清理仍在运行的进程
        SGLANG_PIDS=$(pgrep -f "sglang.launch_server" 2>/dev/null || true)
        if [ -n "${SGLANG_PIDS}" ]; then
            for pid in ${SGLANG_PIDS}; do
                if kill -0 "${pid}" 2>/dev/null; then
                    echo "  强制清理进程 ${pid}..."
                    kill -9 "${pid}" 2>/dev/null || true
                fi
            done
        fi
    fi
    
    # 方法3: 通过端口清理所有相关进程
    echo "检查端口 ${PORT} 上的进程..."
    PORT_PIDS=$(ss -ltnp 2>/dev/null | awk -v p=":${PORT}" '$4~p {split($7,a,"[,=]"); print a[2]}' | sort -u || true)
    if [ -n "${PORT_PIDS}" ]; then
        for pid in ${PORT_PIDS}; do
            if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
                echo "  清理占用端口 ${PORT} 的进程 ${pid}..."
                kill "${pid}" 2>/dev/null || true
                sleep 1
                if kill -0 "${pid}" 2>/dev/null; then
                    kill -9 "${pid}" 2>/dev/null || true
                fi
            fi
        done
    fi
    
    # 等待一下确保进程完全退出
    sleep 1
    
    # 最终检查
    REMAINING_PIDS=$(pgrep -f "sglang.launch_server" 2>/dev/null || true)
    if [ -n "${REMAINING_PIDS}" ]; then
        REMAINING_COUNT=$(echo "${REMAINING_PIDS}" | wc -l)
        echo "警告: 仍有 ${REMAINING_COUNT} 个 sglang 进程在运行:"
        for pid in ${REMAINING_PIDS}; do
            ps -p "${pid}" -o pid,cmd --no-headers 2>/dev/null || true
        done
    else
        echo "✓ 所有 sglang 进程已清理"
    fi
    
    echo "清理完成"
}
trap cleanup EXIT INT TERM

echo ""
echo "========== 步骤 2: 运行验证 =========="

# 模型名：优先使用脚本参数传入的完整模型名；否则从服务器获取完整模型 ID
if [ -n "${MODEL_NAME_OVERRIDE}" ]; then
    MODEL_NAME="${MODEL_NAME_OVERRIDE}"
else
    _MODELS_URL="http://${API_HOST}:${PORT}/v1/models"
    SERVER_MODEL_ID="$(curl -s "${_MODELS_URL}" | python -c \
'import json,sys
d=json.load(sys.stdin)
data=d.get("data") or []
print(data[0].get("id","") if data and isinstance(data,list) and isinstance(data[0],dict) else "")
' 2>/dev/null)"
    MODEL_NAME="${SERVER_MODEL_ID:-${BASE_MODEL}}"
fi
echo "使用模型ID: ${MODEL_NAME}"

# 运行验证（与 scripts/validation/val.sh 保持一致）
VALIDATION_CMD="python -m src.practices.article.validation"
VALIDATION_CMD="${VALIDATION_CMD} --model \"${MODEL_NAME}\""
VALIDATION_CMD="${VALIDATION_CMD} --provider \"${PROVIDER}\""
VALIDATION_CMD="${VALIDATION_CMD} --thinking_model ${THINKING_MODEL}"
VALIDATION_CMD="${VALIDATION_CMD} --enable_thinking ${ENABLE_THINKING}"
VALIDATION_CMD="${VALIDATION_CMD} --enable_logp ${ENABLE_LOGP}"
VALIDATION_CMD="${VALIDATION_CMD} ${AVG_N:+--avg_n ${AVG_N}}"
VALIDATION_CMD="${VALIDATION_CMD} --max_concurrent ${MAX_CONCURRENT}"
VALIDATION_CMD="${VALIDATION_CMD} --split validate"
VALIDATION_CMD="${VALIDATION_CMD} --prompt \"${PROMPT}\""
VALIDATION_CMD="${VALIDATION_CMD} --subjects \"${SUBJECT}\""
VALIDATION_CMD="${VALIDATION_CMD} ${TYPES:+--types ${TYPES}}"
VALIDATION_CMD="${VALIDATION_CMD} --entry rq_with_context"
VALIDATION_CMD="${VALIDATION_CMD} --data_source ${DATA_SOURCE}"
VALIDATION_CMD="${VALIDATION_CMD} --config_path ./assets/model.toml"
VALIDATION_CMD="${VALIDATION_CMD} --temperature ${TEMPERATURE}"
VALIDATION_CMD="${VALIDATION_CMD} --output_dir ./outcome/${PROMPT}_${SUBJECT}"
VALIDATION_CMD="${VALIDATION_CMD} --overwrite_cache"

# 根据数据源类型添加相应参数
if [ "${DATA_SOURCE}" = "jsonl" ]; then
    VALIDATION_CMD="${VALIDATION_CMD} --jsonl_path \"${JSONL_PATH}\""
elif [ "${DATA_SOURCE}" = "mongodb" ]; then
    VALIDATION_CMD="${VALIDATION_CMD} --db_name ${DB_NAME}"
    # 如果提供了 MongoDB 连接字符串，则使用它
    if [ -n "${MONGODB_CONNECTION_STRING}" ]; then
        VALIDATION_CMD="${VALIDATION_CMD} --connection_string \"${MONGODB_CONNECTION_STRING}\""
    fi
fi

echo "执行验证命令:"
echo "${VALIDATION_CMD}"
echo ""

eval "${VALIDATION_CMD}"

echo ""
echo "========== 验证完成 =========="

# 主动清理 sglang（即使 trap 也会清理，这里提前确保收尾）
cleanup
exit 0

