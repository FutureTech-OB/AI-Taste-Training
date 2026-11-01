"""
Article SFT 训练 CLI 入口

调用路径: article → dataloader → messages → trainer

# OpenAI + JSONL
    python -m src.practices.article.sft \\
        --trainer openai \\
        --model gpt-4o-mini-2024-07-18 \\
        --data_source jsonl --jsonl_path ./data/train.jsonl \\
        --prompt ob_rqcontext_simple --entry rq_with_context \\
        --batch_size 32 --epochs 5 \\
        --output_dir ./finetune/ob_rqcontext_simple_ob

# DeepSpeed + JSONL
    python -m src.practices.article.sft \\
        --trainer deepspeed \\
        --model_path /path/to/Qwen3-30B \\
        --data_source jsonl --jsonl_path ./data/train.jsonl \\
        --prompt ob_rqcontext_simple --entry rq_with_context \\
        --batch_size 32 --per_device_batch_size 1 --epochs 5 --lr 1e-4 \\
        --deepspeed_config assets/ds_config_zero3.json \\
        --output_dir ./finetune/ob_rqcontext_simple_ob_ds
"""
import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import AsyncIterator, Dict, Any

from src.core.schema.training import TrainingConfig
from src.practices.article.schema import ArticleFilter
from src.practices.article.transformer import ArticleDataTransformer

import src.practices.article  # noqa: F401

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Article SFT 训练 CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 训练器选择 ─────────────────────────────────────────────────────
    parser.add_argument(
        "--trainer", default="deepspeed", choices=["openai", "deepspeed"],
        help="训练器类型",
    )

    # ── 数据源（参考 validation 实现） ─────────────────────────────────
    parser.add_argument(
        "--data_source", default="jsonl", choices=["mongodb", "jsonl"],
        help="数据来源",
    )
    parser.add_argument("--jsonl_path", default=None, help="训练集 JSONL 文件路径（data_source=jsonl 时必填）")
    parser.add_argument("--eval_jsonl_path", default=None, help="验证集 JSONL 文件路径（data_source=jsonl 时可选）")
    parser.add_argument("--db_name", default="RQ", help="MongoDB 数据库名称")
    parser.add_argument("--connection_string", default=None, help="MongoDB 连接字符串")

    # ── 数据过滤 / 转换（两种数据源共用） ─────────────────────────────
    parser.add_argument("--split", default="train", help="数据集划分（train/val/test）")
    parser.add_argument("--subjects", nargs="*", default=None, help="学科列表（如 ob hm）")
    parser.add_argument("--types", nargs="*", default=None, help="文章类型过滤")
    parser.add_argument("--years", nargs="*", type=int, default=None, help="年份列表（如 2021 2022）")
    parser.add_argument("--prompt", required=True, help="Prompt 名称")
    parser.add_argument("--entry", default="rq_with_context", help="数据 entry 字段")
    parser.add_argument("--target_field", default="rank", help="目标标签字段")
    parser.add_argument("--output_dir", default="./finetune", help="训练文件 / 模型输出目录")
    parser.add_argument("--batch_size", type=int, default=16, help="全局 batch size")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--balance", type=int, default=None, metavar="N",
                        help="按类别平衡：每类最多 N 条（不足则全取），用 seed 随机采样；不传则不 balance")
    parser.add_argument("--balance_seed", type=int, default=42, help="balance 时的随机种子")
    parser.add_argument("--balance_strategy", default="random", choices=["random", "year_desc"],
                        help="balance 采样策略：random 随机；year_desc 优先选择最新年份")

    # ── OpenAI 专有 ────────────────────────────────────────────────────
    openai_group = parser.add_argument_group("OpenAI")
    openai_group.add_argument("--model", default=None, help="OpenAI 基础模型名称")
    openai_group.add_argument("--config_path", default="assets/model.toml")
    openai_group.add_argument("--provider", default="openai.official")
    openai_group.add_argument("--lr_mult", type=float, default=1.0, help="学习率乘数")
    openai_group.add_argument("--n_checkpoints", type=int, default=1)
    openai_group.add_argument("--suffix", default=None, help="微调模型名后缀（≤18 字符）")

    # ── DeepSpeed 专有 ─────────────────────────────────────────────────
    ds_group = parser.add_argument_group("DeepSpeed")
    ds_group.add_argument("--model_path", default=None, help="本地模型权重路径")
    ds_group.add_argument("--lr", type=float, default=1e-4, help="学习率")
    ds_group.add_argument("--per_device_batch_size", type=int, default=1)
    ds_group.add_argument("--max_length", type=int, default=4096)
    ds_group.add_argument("--deepspeed_config", default=None, help="DeepSpeed JSON 配置路径")
    ds_group.add_argument("--gradient_checkpointing", action="store_true")
    ds_group.add_argument("--save_steps", type=int, default=100, help="save_strategy=steps 时每 N 步存盘")
    ds_group.add_argument("--save_strategy", default="epoch", choices=["steps", "epoch"],
                          help="steps=按 save_steps 存盘；epoch=每 epoch 结束时存盘")
    ds_group.add_argument("--save_only_model", action="store_true", default=True,
                          help="仅保存模型，不保存优化器/调度器/rng（默认 True）")
    ds_group.add_argument("--no-save_only_model", action="store_false", dest="save_only_model",
                          help="关闭 save_only_model，保存完整 checkpoint 以支持断点续训")
    ds_group.add_argument("--eval_strategy", default="epoch", choices=["steps", "epoch"],
                          help="验证策略（有验证集时生效）")
    ds_group.add_argument("--master_port", type=int, default=28500, help="分布式 rendezvous 端口")
    ds_group.add_argument("--attn_implementation", default=None)
    ds_group.add_argument("--use_peft", action="store_true")
    ds_group.add_argument("--lora_r", type=int, default=16)
    ds_group.add_argument("--lora_alpha", type=int, default=32)

    return parser.parse_args()


def _validate_args(args):
    if args.trainer == "openai" and not args.model:
        raise ValueError("--trainer openai 需要指定 --model")
    if args.trainer == "deepspeed" and not args.model_path:
        raise ValueError("--trainer deepspeed 需要指定 --model_path")


async def _stream_from_jsonl(path: str) -> AsyncIterator[Dict[str, Any]]:
    """从 JSONL 文件逐行读取，yield 每行解析后的 dict（与 Article 导出结构一致）。"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


async def main():
    args = parse_args()
    _validate_args(args)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")
    logging.getLogger("__main__").setLevel(logging.INFO)
    logging.getLogger("src.core.sft").setLevel(logging.INFO)
    logging.getLogger("src.practices.article.transformer").setLevel(logging.INFO)

    # ── 数据加载器（参考 validation 实现） ────────────────────────────
    article_filter = ArticleFilter(
        split=args.split,
        subjects=args.subjects,
        types=args.types,
        years=args.years,
    )

    loader, data_source_label, job_class = await _init_data_source(args)

    total = await loader.count(article_filter)

    data_config = {
        "entry": args.entry,
        "prompt_name": args.prompt,
        "target_field": args.target_field,
        "balance_max_per_class": getattr(args, "balance", None),
        "balance_seed": getattr(args, "balance_seed", 42),
        "balance_strategy": getattr(args, "balance_strategy", "random"),
    }

    def transform_func(stream, cfg):
        return ArticleDataTransformer.transform_stream(
            stream,
            entry=cfg["entry"],
            prompt_name=cfg["prompt_name"],
            target_field=cfg["target_field"],
            balance_max_per_class=cfg.get("balance_max_per_class"),
            balance_seed=cfg.get("balance_seed", 42),
            balance_strategy=cfg.get("balance_strategy", "random"),
        )

    data_filter_dict = {
        "split": args.split,
        "subjects": args.subjects,
        "types": args.types,
        "years": args.years,
    }

    # ── 验证集 ────────────────────────────────────────────────────────
    eval_stream = None
    eval_total = 0
    if args.trainer == "deepspeed":
        if args.eval_jsonl_path:
            # 指定 JSONL 作 eval：从文件读入，走同一套 transform
            eval_stream = _stream_from_jsonl(args.eval_jsonl_path)
            eval_total = -1
        elif args.data_source == "mongodb":
            eval_filter = ArticleFilter(split="validate", subjects=args.subjects, types=args.types, years=args.years)
            eval_total = await loader.count(eval_filter)
            if eval_total > 0:
                eval_stream = loader.load_stream(eval_filter)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── trainer 分支 ───────────────────────────────────────────────────
    if args.trainer == "openai":
        trainer, config = _build_openai(args, job_class)
    else:
        trainer, config = _build_deepspeed(args, job_class)

    # ── 摘要 ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  trainer  : {args.trainer}")
    if args.trainer == "openai":
        print(f"  model    : {args.model}  suffix={args.suffix}")
        print(f"  lr_mult  : {args.lr_mult}")
    else:
        print(f"  model    : {Path(args.model_path).name}")
        print(f"  lr       : {args.lr}  per_device_bs={args.per_device_batch_size}")
    print(f"  数据源   : {data_source_label}")
    print(f"  filter   : split={args.split}  subjects={args.subjects}  types={args.types}  years={args.years}  entry={args.entry}")
    print(f"  查询命中 : {total} 条（无 entry/无 label 的会在建数据集时筛掉，实际样本数见下方 [transform] 与 samples）")
    if eval_total > 0:
        print(f"  eval     : split=validate  命中={eval_total} 条")
    elif eval_total == -1:
        print(f"  eval     : {args.eval_jsonl_path}")
    print(f"  prompt   : {args.prompt}  entry={args.entry}")
    if getattr(args, "balance", None) is not None:
        print(f"  balance  : max_per_class={args.balance}  seed={getattr(args, 'balance_seed', 42)}")
    print(f"  epochs={args.epochs}  batch={args.batch_size}")
    print(f"{'='*60}\n")

    job = await trainer.train(
        data_stream=loader.load_stream(article_filter),
        config=config,
        practice="article",
        data_filter=data_filter_dict,
        data_config=data_config,
        transform_func=transform_func,
        eval_stream=eval_stream,
        eval_transform_func=transform_func if eval_stream is not None else None,
    )

    print(f"\n{'='*60}")
    print(f"  训练完成!")
    print(f"  samples  : {job.training_samples}")
    print(f"  job_id   : {job.job_id}")
    print(f"  model    : {job.fine_tuned_model}")
    print(f"  file     : {job.train_file_path}")
    print(f"{'='*60}\n")


async def _init_data_source(args):
    """初始化数据加载器；job_class 由 core.sft.setup.init_job_tracking 统一决定。"""
    from src.core.sft.setup import init_job_tracking

    if args.data_source == "jsonl":
        from src.core.dataloader.jsonl import JSONLLoader
        loader = JSONLLoader(args.jsonl_path, id_field="title", flush_interval=0)
        label = f"JSONL ({args.jsonl_path})"
        # 仅当用户明确指定了非默认 db 配置时才连 MongoDB
        has_db = bool(args.connection_string or args.db_name != "RQ")
        job_class = await init_job_tracking(
            db_name=args.db_name if has_db else None,
            connection_string=args.connection_string if has_db else None,
        )
    else:
        from src.practices.article.models.article import Article
        from src.practices.article.models.pdf import PDFData
        from src.practices.article.loader import ArticleLoader
        loader = ArticleLoader()
        label = f"MongoDB ({args.db_name})"
        job_class = await init_job_tracking(
            db_name=args.db_name,
            connection_string=args.connection_string,
            extra_document_models=[Article, PDFData],
        )

    logger.info("数据源: %s", label)
    return loader, label, job_class


def _build_openai(args, job_class):
    from src.core.utils.config import ConfigLoader
    from src.core.sft.openai import OpenAITrainer

    provider_config = ConfigLoader.get_provider_config_with_env_fallback(
        args.config_path, args.provider
    )
    api_key = provider_config.get("api_key")
    if not api_key:
        raise ValueError(
            f"无法从 {args.config_path} 获取 provider={args.provider} 的 API key"
        )

    # OpenAI API 的 suffix 最多 18 字符；文件名用完整 suffix 便于识别
    suffix_api = args.suffix[:18] if args.suffix else None
    suffix_full = (args.suffix or "").strip() or None
    config = TrainingConfig(
        model=args.model,
        model_name=f"{args.model}__{suffix_full}" if suffix_full else None,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr_mult,
        output_dir=args.output_dir,
        platform_config={"suffix": suffix_api, "n_checkpoints": args.n_checkpoints},
    )
    return OpenAITrainer(api_key=api_key, job_class=job_class), config


def _build_deepspeed(args, job_class):
    from src.core.sft.deepspeed import DeepSpeedTrainer
    from src.core.sft.deepspeed.config import DeepSpeedSFTConfig

    config = DeepSpeedSFTConfig(
        model=Path(args.model_path).name,
        model_path=args.model_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        per_device_batch_size=args.per_device_batch_size,
        max_length=args.max_length,
        gradient_checkpointing=args.gradient_checkpointing,
        save_steps=getattr(args, "save_steps", 100),
        save_strategy=getattr(args, "save_strategy", "epoch"),
        save_only_model=getattr(args, "save_only_model", True),
        deepspeed_config=args.deepspeed_config,
        eval_file=getattr(args, "eval_jsonl_path", None),
        eval_strategy=getattr(args, "eval_strategy", "epoch"),
        master_port=args.master_port,
        attn_implementation=args.attn_implementation,
        use_peft=args.use_peft,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    return DeepSpeedTrainer(config, job_class=job_class), config


if __name__ == "__main__":
    asyncio.run(main())
