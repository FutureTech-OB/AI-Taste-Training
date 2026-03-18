#!/bin/bash
set -e

# mergekit SLERP 合并脚本
# SLERP (Spherical Linear Interpolation) 在两个模型之间进行球面线性插值
# 用法：
#   chmod +x mergekit_slerp.sh
#   ./mergekit_slerp.sh

# ========== 配置参数（请根据实际情况修改） ==========
# 输出目录
OUTPUT_DIR="./merged_models/slerp_merge"

# 是否使用 CUDA
USE_CUDA=true

# 是否使用延迟加载（节省内存）
LAZY_UNPICKLE=false

# 是否允许不兼容架构合并
ALLOW_CRIMES=false

# 输出分片大小
OUT_SHARD_SIZE="5B"

# ========== YAML 配置内容（请根据实际情况修改） ==========
CONFIG_YAML=$(cat <<'EOF'
models:
  - model: model_a_path
  - model: model_b_path
merge_method: slerp
base_model: model_a_path
parameters:
  t: 0.5
dtype: float16
EOF
)

# ========== 创建临时配置文件 ==========
TMP_CONFIG_FILE=$(mktemp /tmp/mergekit_config_XXXXXX.yml)
echo "${CONFIG_YAML}" > "${TMP_CONFIG_FILE}"

# 清理函数：退出时删除临时文件
cleanup() {
    rm -f "${TMP_CONFIG_FILE}"
}
trap cleanup EXIT

echo "========== mergekit SLERP 合并 =========="
echo "临时配置文件: ${TMP_CONFIG_FILE}"
echo "输出目录: ${OUTPUT_DIR}"
echo "使用 CUDA: ${USE_CUDA}"
echo ""
echo "配置内容:"
echo "${CONFIG_YAML}"
echo "======================================="
echo ""

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# ========== 构建命令 ==========
MERGE_CMD="mergekit-yaml"
MERGE_CMD="${MERGE_CMD} \"${TMP_CONFIG_FILE}\""
MERGE_CMD="${MERGE_CMD} \"${OUTPUT_DIR}\""

if [ "${USE_CUDA}" = "true" ]; then
    MERGE_CMD="${MERGE_CMD} --cuda"
fi

if [ "${LAZY_UNPICKLE}" = "true" ]; then
    MERGE_CMD="${MERGE_CMD} --lazy-unpickle"
fi

if [ "${ALLOW_CRIMES}" = "true" ]; then
    MERGE_CMD="${MERGE_CMD} --allow-crimes"
fi

if [ -n "${OUT_SHARD_SIZE}" ]; then
    MERGE_CMD="${MERGE_CMD} --out-shard-size ${OUT_SHARD_SIZE}"
fi

# ========== 执行合并 ==========
echo "执行命令:"
echo "${MERGE_CMD}"
echo ""

eval "${MERGE_CMD}"

if [ $? -eq 0 ]; then
    echo ""
    echo "========== 合并完成 =========="
    echo "合并后的模型已保存到: ${OUTPUT_DIR}"
    echo ""
else
    echo ""
    echo "错误: 合并失败"
    exit 1
fi
