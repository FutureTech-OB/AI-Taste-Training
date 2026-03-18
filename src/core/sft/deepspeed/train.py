"""
TRL SFT + DeepSpeed 训练脚本

参数风格与 deepspeed_sft.sh 保持一致（短名、总 batch size 自动推算 grad_accum）：

    deepspeed --num_gpus 8 src/core/sft/deepspeed/train.py \\
        --model_path /path/to/model \\
        --data_file  /path/to/train.jsonl \\
        --output_dir /path/to/output \\
        --per_device_batch_size 1 \\
        --batch_size 32 \\
        --epochs 5 \\
        --lr 1e-4 \\
        --max_length 4096 \\
        --save_steps 100 \\
        --gradient_checkpointing \\
        --deepspeed assets/ds_config_zero3.json

--batch_size 为总 batch size（= per_device_batch_size × world_size × grad_accum）。
若同时传入 --gradient_accumulation_steps，则以其为准，忽略 --batch_size。

数据格式（JSONL 每行一条）:
  conversational : {"messages": [{"role": "user", "content": "..."}, ...]}
  language model : {"text": "..."}
"""
import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)


# ── 参数解析 ──────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DeepSpeed + TRL SFT 训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 数据 ──────────────────────────────────────────────────────────
    p.add_argument("--data_file", required=True, help="训练集 JSONL 文件路径")
    p.add_argument("--eval_file", default=None, help="验证集 JSONL 文件路径（可选）")
    p.add_argument("--dataset_text_field", default="text", help="JSONL 文本字段名")

    # ── 模型 ──────────────────────────────────────────────────────────
    p.add_argument("--model_path", required=True, help="本地模型权重路径")
    p.add_argument("--attn_implementation", default=None,
                   help="注意力实现：flash_attention_2 / sdpa / eager")
    p.add_argument("--trust_remote_code", action="store_true")

    # ── 训练超参 ──────────────────────────────────────────────────────
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--per_device_batch_size", type=int, default=1, metavar="N",
                   help="每卡 micro batch size")
    p.add_argument("--batch_size", type=int, default=None, metavar="N",
                   help="全局 batch size；自动推算 gradient_accumulation_steps")
    p.add_argument("--gradient_accumulation_steps", type=int, default=None,
                   help="显式指定 grad accum（优先于 --batch_size）")
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--pad_to_multiple_of", type=int, default=512,
                   help="序列长度对齐到该值的倍数（提升硬件利用率）")
    p.add_argument("--completion_only_loss", action=argparse.BooleanOptionalAction, default=True,
                   help="只对 assistant 回复部分计算 loss（默认开启）")
    p.add_argument("--packing", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True,
                   help="启用 bf16（默认开启；--no-bf16 可关闭）")
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--save_strategy", default="steps", choices=["steps", "epoch"],
                   help="steps=按 save_steps 存盘；epoch=每 epoch 结束时存盘")
    p.add_argument("--save_only_model", action="store_true", default=True,
                   help="仅保存模型权重，不保存优化器/调度器/rng（默认 True）")
    p.add_argument("--no-save_only_model", action="store_false", dest="save_only_model",
                   help="关闭 save_only_model，保存完整 checkpoint 以支持断点续训")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_strategy", default="no",
                   choices=["no", "steps", "epoch"])

    # ── DeepSpeed ─────────────────────────────────────────────────────
    p.add_argument("--deepspeed", default=None, help="DeepSpeed JSON 配置文件路径")

    # ── PEFT / LoRA ───────────────────────────────────────────────────
    p.add_argument("--use_peft", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", nargs="+", default=None)

    return p


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def resolve_grad_accum(args: argparse.Namespace) -> int:
    """
    优先级：--gradient_accumulation_steps > 从 --batch_size 推算 > 默认值 8。
    world_size 通过 WORLD_SIZE 环境变量（deepspeed/accelerate 自动设置）获取。
    """
    if args.gradient_accumulation_steps is not None:
        return args.gradient_accumulation_steps
    if args.batch_size is not None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        return max(1, args.batch_size // (args.per_device_batch_size * world_size))
    return 8


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    # parse_known_args：忽略 deepspeed/accelerate 注入的 --local_rank 等内部参数
    args, _ = build_parser().parse_known_args()
    grad_accum = resolve_grad_accum(args)
    logger.info("gradient_accumulation_steps = %d", grad_accum)

    # ── Tokenizer ─────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 模型初始化参数 ─────────────────────────────────────────────────
    # 传字符串路径给 SFTTrainer，由 TRL 在 deepspeed.zero.Init() 上下文中完成
    # from_pretrained，实现 ZeRO-3 下参数分片即时初始化，避免每卡加载完整模型。
    model_init_kwargs: dict = {
        "dtype": torch.bfloat16 if args.bf16 else torch.float32,
        "low_cpu_mem_usage": True,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_init_kwargs["attn_implementation"] = args.attn_implementation

    # ── 数据集 ────────────────────────────────────────────────────────
    data_files: dict = {"train": args.data_file}
    if args.eval_file:
        data_files["test"] = args.eval_file
    dataset = load_dataset("json", data_files=data_files)

    # completion_only_loss：将 messages 格式在线转为 prompt-completion 格式，
    # 用 TRL 0.29.0 原生的 completion_only_loss=True 机制，不依赖 chat template
    # 里是否有 {% generation %} 标记（Qwen3 没有）。
    if args.completion_only_loss and "messages" in dataset["train"].column_names:
        def _to_prompt_completion(examples, _tok=tokenizer):
            prompts, completions = [], []
            for msgs in examples["messages"]:
                prompts.append(_tok.apply_chat_template(
                    msgs[:-1], tokenize=False, add_generation_prompt=True
                ))
                completions.append(msgs[-1]["content"])
            return {"prompt": prompts, "completion": completions}

        dataset = dataset.map(_to_prompt_completion, batched=True, remove_columns=["messages"])
        logger.info("已转为 prompt-completion 格式，启用 completion_only_loss")

    eval_dataset = (
        dataset["test"]
        if args.eval_file and args.eval_strategy != "no"
        else None
    )

    # ── SFTConfig ─────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=args.max_length,
        pad_to_multiple_of=args.pad_to_multiple_of,
        completion_only_loss=args.completion_only_loss,
        packing=args.packing,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        save_only_model=args.save_only_model,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        dataset_text_field=args.dataset_text_field,
        deepspeed=args.deepspeed,
        model_init_kwargs=model_init_kwargs,
    )

    # ── PEFT / LoRA ───────────────────────────────────────────────────
    peft_config = None
    if args.use_peft:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            task_type="CAUSAL_LM",
        )

    # ── 训练 ──────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=args.model_path,          # 传路径字符串；TRL 在 ZeRO-3 init 上下文中加载
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    final_dir = str(Path(args.output_dir) / "final_model")
    trainer.save_model(final_dir)
    logger.info("Model saved to %s", final_dir)


if __name__ == "__main__":
    main()
