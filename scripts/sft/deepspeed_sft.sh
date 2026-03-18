#!/bin/bash
set -e

# ── 数据配置 ────────────────────────────────────────────────────────────────
# 训练前需先将数据导出为 JSONL（通过 DeepSpeedTrainer.prepare_training_data 或手动导出）
DATA_FILE="/workspace/gongziqin/RQ/data/train.jsonl"
# EVAL_FILE="/workspace/gongziqin/RQ/data/eval.jsonl"  # 可选

# ── 模型配置 ────────────────────────────────────────────────────────────────
MODEL_PATH="/workspace/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-Instruct-2507"

# ── 训练配置 ────────────────────────────────────────────────────────────────
OUT_DIR="./finetune/run_ds"

PER_DEVICE_BATCH_SIZE=1
BATCH_SIZE=32          # 全局 batch size；train.py 自动推算 gradient_accumulation_steps

# ── GPU 数量（Qwen3-30B-A3B MoE 需要 8 卡，EP=8，每卡持有 16/128 个 expert） ──
NUM_GPUS=8

mkdir -p "${OUT_DIR}"

export PYTHONPATH="/workspace/gongziqin/228/RQ:${PYTHONPATH}"
MASTER_PORT=28500

deepspeed --num_gpus ${NUM_GPUS} --master_port ${MASTER_PORT} \
  src/core/sft/deepspeed/train.py \
  --model_path "${MODEL_PATH}" \
  --data_file "${DATA_FILE}" \
  --per_device_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --batch_size ${BATCH_SIZE} \
  --epochs 5 \
  --lr 2e-5 \
  --max_length 4096 \
  --save_steps 100 \
  --gradient_checkpointing \
  --deepspeed /workspace/gongziqin/228/RQ/assets/ds_config_zero3.json \
  --output_dir "${OUT_DIR}"

# LoRA 示例（去掉注释即可）：
#   --use_peft \
#   --lora_r 16 \
#   --lora_alpha 32 \

# 加验证集（去掉注释即可）：
#   --eval_file "${EVAL_FILE}" \
#   --eval_strategy steps \
