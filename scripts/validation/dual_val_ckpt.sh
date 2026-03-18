#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/scripts/common/load_db_config.sh"
# 使用 checkpoint 运行 Pairwise (dual) 验证：先启动 SGLang 加载 checkpoint，再跑 dual_validation。
# 用法: ./dual_val_ckpt.sh [MODEL_NAME] [TEMPERATURE] [MAX_CONCURRENT] [CLEANUP_SGLANG]
#   MODEL_NAME: 可选，不传则从 /v1/models 读取
#   TEMPERATURE / MAX_CONCURRENT: 可选
#   CLEANUP_SGLANG: 1/yes=启动时 kill 并启动 SGLang，结束时释放；0/no=不 kill 不启动（假定已在运行），结束时也不释放

set -e

# ========== Checkpoint 与 SGLang ==========
CHECKPOINT_PATH="/workspace/gongziqin/228/RQ/finetune/ob_rqcontext_ob_30B_/checkpoint-103"
PORT=8000
HOST="0.0.0.0"
TP_SIZE=2
DP_SIZE=4
PP_SIZE=1
EP_SIZE=2
NNODES=1
NODE_RANK=0
DIST_INIT_ADDR=""
BASE_MODEL="qwen3-30b-a3b"
PROVIDER="openai.sglang"

MODEL_NAME_OVERRIDE="${1:-}"
TEMPERATURE="${2:-1.0}"
MAX_CONCURRENT="${3:-10}"
THINKING_MODEL="${4:-True}"
ENABLE_THINKING="${5:-True}"
# 1=启动时 kill 并启动 SGLang，结束时释放；0=不 kill 不启动、结束时也不释放（假定 SGLang 已在运行）
CLEANUP_SGLANG_RAW="${6:-1}"
case "${CLEANUP_SGLANG_RAW}" in
    0|no|false|n) CLEANUP_SGLANG="false" ;;
    *)            CLEANUP_SGLANG="true" ;;
esac

# ========== Dual 任务配置（与 dual_val.sh 一致）==========
PROMPT="ob_rqcontext_dual"
ENTRY="rq_with_context"
SUBJECT="ob"
TYPES="study"
SEED=42
DATA_SOURCE="jsonl"
JSONL_PATH="/workspace/gongziqin/228/RQ/assets/unknown/120.jsonl"
DB_NAME="RQ0"
load_db_config "${REPO_ROOT}" "${DB_NAME}" || exit 1
CONNECTION_STRING="${MONGODB_CONNECTION_STRING}"
SPLIT="validate"
OUT_DIR="./outcome/dual_${PROMPT}_${SUBJECT}"

# ========== 公共：model.toml + 等待服务就绪 ==========
mkdir -p ./assets
API_HOST="${HOST}"
[ "${HOST}" = "0.0.0.0" ] && API_HOST="localhost"

MODEL_TOML="./assets/model.toml"
if [ -f "${MODEL_TOML}" ]; then
    grep -q "^\[providers\.openai\.sglang\]" "${MODEL_TOML}" || {
        echo "" >> "${MODEL_TOML}"
        echo "[providers.openai.sglang]" >> "${MODEL_TOML}"
        echo "api_key = \"sk-0\"" >> "${MODEL_TOML}"
        echo "base_url = \"http://${API_HOST}:${PORT}/v1\"" >> "${MODEL_TOML}"
        echo "已追加配置到 ${MODEL_TOML}"
    }
else
    cat > "${MODEL_TOML}" <<EOF
[providers.openai.sglang]
api_key = "sk-0"
base_url = "http://${API_HOST}:${PORT}/v1"
EOF
    echo "已创建配置文件 ${MODEL_TOML}"
fi

if [ "${CLEANUP_SGLANG}" = "true" ]; then
    # ---------- 启动前检查：是否已安装 sglang ----------
    if ! python -c "import sglang" 2>/dev/null; then
        echo "错误: 未检测到 sglang，无法启动服务。请先安装: pip install sglang[all]"
        exit 1
    fi
    if ! python -c "import sglang.launch_server" 2>/dev/null; then
        echo "错误: 未检测到 sglang.launch_server，请确认 sglang 安装完整（如 pip install sglang[all]）"
        exit 1
    fi
    echo "✓ sglang 已安装"

    # ---------- 启动时 kill 并启动 SGLang ----------
    kill_process_on_port() {
        local port=$1
        EXIST_PID=$(ss -ltnp 2>/dev/null | awk -v p=":${port}" '$4~p {split($7,a,"[,=]"); print a[2]; exit}')
        if [ -n "${EXIST_PID}" ]; then
            echo "检测到端口 ${port} 已被进程 ${EXIST_PID} 占用，尝试结束..."
            kill "${EXIST_PID}" 2>/dev/null || true
            sleep 2
            if kill -0 "${EXIST_PID}" 2>/dev/null; then
                kill -9 "${EXIST_PID}" 2>/dev/null || true
                sleep 1
            fi
            EXIST_PID=$(ss -ltnp 2>/dev/null | awk -v p=":${port}" '$4~p {split($7,a,"[,=]"); print a[2]; exit}')
            [ -z "${EXIST_PID}" ] && echo "✓ 端口 ${port} 已释放" || echo "警告: 端口 ${port} 仍被占用"
        else
            echo "端口 ${port} 未被占用"
        fi
    }
    echo "========== 检查端口 ${PORT} =========="
    kill_process_on_port ${PORT}
    echo ""

    if [ ! -d "${CHECKPOINT_PATH}" ]; then
        echo "错误: checkpoint路径不存在: ${CHECKPOINT_PATH}"
        exit 1
    fi

    echo "========== 步骤 1: 启动 SGLang =========="
    echo "Checkpoint: ${CHECKPOINT_PATH}  端口: ${PORT}  结束后释放: ${CLEANUP_SGLANG}"

    kill_sglang_on_port() {
        EXIST_PID=$(ss -ltnp 2>/dev/null | awk -v p=":${PORT}" '$4~p {split($7,a,"[,=]"); print a[2]; exit}')
        if [ -n "${EXIST_PID}" ]; then
            kill "${EXIST_PID}" 2>/dev/null || true
            sleep 1
            kill -9 "${EXIST_PID}" 2>/dev/null || true
        fi
    }
    kill_sglang_on_port

    SGLANG_CMD="python -m sglang.launch_server"
    SGLANG_CMD="${SGLANG_CMD} --model-path \"${CHECKPOINT_PATH}\" --port ${PORT} --host \"${HOST}\""
    SGLANG_CMD="${SGLANG_CMD} --trust-remote-code --dtype bfloat16"
    [ "${TP_SIZE}" != "1" ] && SGLANG_CMD="${SGLANG_CMD} --tensor-parallel-size ${TP_SIZE}"
    [ "${DP_SIZE}" != "1" ] && SGLANG_CMD="${SGLANG_CMD} --data-parallel-size ${DP_SIZE}"
    [ "${PP_SIZE}" != "1" ] && SGLANG_CMD="${SGLANG_CMD} --pipeline-parallel-size ${PP_SIZE}"
    [ "${EP_SIZE}" != "1" ] && SGLANG_CMD="${SGLANG_CMD} --expert-parallel-size ${EP_SIZE}"
    [ "${NNODES}" != "1" ] && SGLANG_CMD="${SGLANG_CMD} --nnodes ${NNODES} --node-rank ${NODE_RANK}"
    [ -n "${DIST_INIT_ADDR}" ] && SGLANG_CMD="${SGLANG_CMD} --dist-init-addr \"${DIST_INIT_ADDR}\""

    echo "执行: ${SGLANG_CMD}"
    eval "${SGLANG_CMD}" > sglang_server_dual.log 2>&1 &
    SGLANG_PID=$!
    echo "SGLang PID: ${SGLANG_PID}  日志: sglang_server_dual.log"

    echo "等待服务器启动..."
    sleep 5
    TIMEOUT_ITERATIONS=300
    i=1
    while [ $i -le ${TIMEOUT_ITERATIONS} ]; do
        if curl -s "http://${API_HOST}:${PORT}/v1/models" > /dev/null 2>&1; then
            echo "✓ 服务器已就绪"
            break
        fi
        if ! kill -0 $SGLANG_PID 2>/dev/null; then
            echo "错误: SGLang 进程已退出"
            tail -50 sglang_server_dual.log
            exit 1
        fi
        [ $i -eq ${TIMEOUT_ITERATIONS} ] && { echo "启动超时"; tail -50 sglang_server_dual.log; kill $SGLANG_PID 2>/dev/null || true; exit 1; }
        echo "等待中... ($i/${TIMEOUT_ITERATIONS})"
        sleep 2
        i=$((i + 1))
    done
else
    # ---------- CLEANUP_SGLANG=0：不 kill 不启动，只等已有服务就绪 ----------
    echo "========== CLEANUP_SGLANG=0：不 kill 不启动 SGLang，等待端口 ${PORT} 已有服务 =========="
    TIMEOUT_ITERATIONS=60
    i=1
    while [ $i -le ${TIMEOUT_ITERATIONS} ]; do
        if curl -s "http://${API_HOST}:${PORT}/v1/models" > /dev/null 2>&1; then
            echo "✓ 服务器已就绪"
            break
        fi
        [ $i -eq ${TIMEOUT_ITERATIONS} ] && { echo "错误: 端口 ${PORT} 上无响应，请先启动 SGLang 或设 CLEANUP_SGLANG=1"; exit 1; }
        echo "等待中... ($i/${TIMEOUT_ITERATIONS})"
        sleep 2
        i=$((i + 1))
    done
fi

cleanup () {
    echo ""
    if [ "${CLEANUP_SGLANG}" != "true" ]; then
        echo "========== 保留 SGLang（CLEANUP_SGLANG=false）=========="
        return 0
    fi
    echo "========== 清理 SGLang =========="
    [ -n "${SGLANG_PID}" ] && kill -0 "${SGLANG_PID}" 2>/dev/null && kill -TERM "${SGLANG_PID}" 2>/dev/null || true
    sleep 2
    for pid in $(pgrep -f "sglang.launch_server" 2>/dev/null); do kill -9 "${pid}" 2>/dev/null || true; done
    for pid in $(ss -ltnp 2>/dev/null | awk -v p=":${PORT}" '$4~p {split($7,a,"[,=]"); print a[2]}' | sort -u); do
        [ -n "${pid}" ] && kill -9 "${pid}" 2>/dev/null || true
    done
    echo "清理完成"
}
trap cleanup EXIT INT TERM

echo ""
echo "========== 步骤 2: Dual (Pairwise) 验证 =========="

if [ -n "${MODEL_NAME_OVERRIDE}" ]; then
    MODEL_NAME="${MODEL_NAME_OVERRIDE}"
else
    MODEL_NAME="$(curl -s "http://${API_HOST}:${PORT}/v1/models" | python -c '
import json,sys
d=json.load(sys.stdin)
data=d.get("data") or []
print(data[0].get("id","") if data and isinstance(data,list) and isinstance(data[0],dict) else "")
' 2>/dev/null)"
    MODEL_NAME="${MODEL_NAME:-${BASE_MODEL}}"
fi
echo "使用模型: ${MODEL_NAME}"

DUAL_CMD="python -m src.practices.article.dual_validation"
DUAL_CMD="${DUAL_CMD} --model \"${MODEL_NAME}\" --provider \"${PROVIDER}\""
DUAL_CMD="${DUAL_CMD} --thinking_model ${THINKING_MODEL} --enable_thinking ${ENABLE_THINKING}"
DUAL_CMD="${DUAL_CMD} --config_path ./assets/model.toml --prompt \"${PROMPT}\" --entry \"${ENTRY}\""
DUAL_CMD="${DUAL_CMD} --subjects \"${SUBJECT}\" ${TYPES:+--types ${TYPES}}"
DUAL_CMD="${DUAL_CMD} --data_source ${DATA_SOURCE} --jsonl_path \"${JSONL_PATH}\""
DUAL_CMD="${DUAL_CMD} --split ${SPLIT} --db_name ${DB_NAME} --connection_string \"${CONNECTION_STRING}\""
DUAL_CMD="${DUAL_CMD} --seed ${SEED} --max_concurrent ${MAX_CONCURRENT} --temperature ${TEMPERATURE}"
DUAL_CMD="${DUAL_CMD} --output_dir \"${OUT_DIR}\""

echo "执行: ${DUAL_CMD}"
eval "${DUAL_CMD}"

echo ""
echo "========== Dual 验证完成 =========="
cleanup
exit 0
