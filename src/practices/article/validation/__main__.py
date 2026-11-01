"""
Article Validation CLI 入口
用法: python -m src.practices.article.validation.validator [args]
      或 python -m src.practices.article.validation [args]
"""
import argparse
import asyncio
import logging
import os

from src.core.utils.config import ConfigLoader
from src.core.utils.sanitize import sanitize_name
from src.core.dataloader.jsonl import JSONLLoader
from src.core.validation.metrics import calculate_metrics, save_results
from src.practices.article.schema import ArticleFilter
from src.practices.article.validation.validator import ArticleValidator

# 确保 article practice 的 prompts 被注册
import src.practices.article  # noqa: F401

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Article Validation")
    parser.add_argument("--model", required=True, help="模型名称（对应 model.toml 中的配置）")
    parser.add_argument("--provider", default=None, help="Provider 名称（如 vertex.official）")
    parser.add_argument("--config_path", default="assets/model.toml", help="配置文件路径")
    parser.add_argument("--prompt", default="ob_rqcontext", help="Prompt 名称")
    parser.add_argument("--entry", default="rq_with_context", help="数据 entry 字段")
    parser.add_argument("--subjects", nargs="*", default=None, help="学科列表")
    parser.add_argument("--types", nargs="*", default=None, help="文章类型列表（如 study review）")
    parser.add_argument("--split", default=None, help="数据集划分 (train/validate/test)")
    parser.add_argument("--enable_logp", type=lambda v: v.lower() in ("true", "1", "yes"), default=True)
    parser.add_argument(
        "--enable_thinking",
        type=lambda v: v.lower() in ("true", "1", "yes"),
        default=False,
        help="启用/允许模型输出推理内容（本项目建议配合 reasoning prompt + enable_logp=False 使用）",
    )
    parser.add_argument(
        "--thinking_model",
        type=lambda v: v.lower() in ("true", "1", "yes"),
        default=False,
        help="是否为“支持 thinking 的模型”（布尔开关）。目前仅用于在 enable_thinking=False 时尝试向部分 provider 发送 thinking=disabled。",
    )
    parser.add_argument("--data_source", choices=["jsonl", "mongodb"], default="jsonl")
    parser.add_argument("--jsonl_path", default=None, help="JSONL 文件路径")
    parser.add_argument("--db_name", default="RQ", help="MongoDB 数据库名称")
    parser.add_argument("--connection_string", default=None, help="MongoDB 连接字符串")
    parser.add_argument("--output_dir", default="./outcome", help="输出目录")
    parser.add_argument("--distill_dir", default=None, help="蒸馏数据目录：每条 request/response 存为独立 JSON 文件（不设则不保存）")
    parser.add_argument("--max_concurrent", type=int, default=10, help="最大并发数")
    parser.add_argument(
        "--avg_n",
        type=int,
        default=None,
        help="文本模式下每条样本重复推理次数（用于多数投票）。不传则默认：thinking_model=True 且 enable_thinking=True 且 enable_logp=False 时为 8，否则为 1。",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度（0=更确定，越大越随机）")
    parser.add_argument("--max_tokens", type=int, default=None, help="最大生成 token 数")
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="覆盖已有缓存：不跳过 val_outcome 中已有结果的样本，全部重新推理并写回 JSONL",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    # ---- 日志 ----
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # ---- 模型 / Provider 配置 ----
    model_config_raw = ConfigLoader.get_model_config(args.config_path, args.model) or {}
    model_config = {
        "model_name": args.model,
        "model_resource": model_config_raw.get("model_resource"),
        "endpoint": model_config_raw.get("endpoint"),
    }

    provider = args.provider or model_config_raw.get("provider")
    if not provider:
        raise ValueError("必须提供 --provider 或确保 model.toml 中的模型配置包含 provider")

    provider_config = ConfigLoader.get_provider_config_with_env_fallback(args.config_path, provider)
    if not provider_config:
        raise ValueError(f"Provider {provider} 配置未找到")

    # ---- 数据加载器 ----
    if args.data_source == "jsonl":
        if not args.jsonl_path:
            raise ValueError("JSONL 模式需要 --jsonl_path")
        dataloader = JSONLLoader(args.jsonl_path, id_field="title", flush_interval=10)
        logger.info(f"数据源: JSONL ({args.jsonl_path})")
    else:
        # MongoDB 模式
        from src.practices.article.models import init_database
        from src.practices.article.loader import ArticleLoader
        conn = args.connection_string or f"mongodb://localhost:27017/{args.db_name}"
        await init_database(conn, db_name=args.db_name)
        dataloader = ArticleLoader()
        logger.info(f"数据源: MongoDB ({args.db_name})")

    # ---- 构建 filter ----
    article_filter = ArticleFilter(
        split=args.split,
        subjects=args.subjects,
        types=args.types,
    )

    safe_model = sanitize_name(args.model)

    # ---- 构建 validator ----
    effective_avg_n = args.avg_n
    if effective_avg_n is None:
        effective_avg_n = 8 if (args.thinking_model and args.enable_thinking and (not args.enable_logp)) else 1
    effective_avg_n = max(1, int(effective_avg_n))
    logger.info(
        f"avg_n (effective) = {effective_avg_n} "
        f"(cli={args.avg_n}, thinking_model={args.thinking_model}, enable_thinking={args.enable_thinking}, enable_logp={args.enable_logp})"
    )

    # 蒸馏目录：按 model 名细分子目录
    distill_dir = None
    if args.distill_dir:
        distill_dir = os.path.join(args.distill_dir, safe_model)
        os.makedirs(distill_dir, exist_ok=True)

    validator = ArticleValidator(
        dataloader=dataloader,
        model_config=model_config,
        provider_config=provider_config,
        entry=args.entry,
        prompt_name=args.prompt,
        enable_logp=args.enable_logp,
        top_logprobs=5,
        max_concurrent=args.max_concurrent,
        avg_n=effective_avg_n,
        thinking_model=args.thinking_model,
        enable_thinking=args.enable_thinking,
        temperature=args.temperature,
        distill_path=distill_dir,
        **({"max_tokens": args.max_tokens} if args.max_tokens is not None else {}),
    )

    # ---- 执行验证 ----
    skip_existing = not args.overwrite_cache
    results = await validator.validate(article_filter, skip_existing=skip_existing)

    # ---- 指标 & 保存 ----
    metrics = calculate_metrics(results)

    output_dir = os.path.join(args.output_dir, safe_model)

    print(f"样本总数: {metrics['total_samples']}")
    if "accuracy_avg" in metrics:
        print(f"平均准确率 (avg_n): {metrics['accuracy_avg']:.2%}")
    if "accuracy_top1" in metrics:
        print(f"Top-1 准确率: {metrics['accuracy_top1']:.2%}")
    if "accuracy_top1_or_top2" in metrics:
        print(f"Top-1+2 命中率: {metrics['accuracy_top1_or_top2']:.2%}")
    if "accuracy_text_match" in metrics:
        print(f"文本匹配准确率: {metrics['accuracy_text_match']:.2%}")
    if "balanced_accuracy" in metrics:
        print(f"Balanced accuracy (macro recall): {metrics['balanced_accuracy']:.2%}")
    if "macro_f1" in metrics:
        print(f"Macro F1: {metrics['macro_f1']:.2%}")

    for label, m in metrics.get("per_label_metrics", {}).items():
        support = m["support"]
        support_str = str(int(round(support))) if isinstance(support, float) else str(support)
        print(f"  {label}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} Support={support_str}")

    save_results(metrics, results, output_dir)
    logger.info("全部完成")


if __name__ == "__main__":
    asyncio.run(main())
