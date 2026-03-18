"""
Dual (Pairwise) Validation CLI.
Usage: python -m src.practices.article.dual_validation [args]
"""
import argparse
import asyncio
import json
import logging
import os

from src.core.utils.config import ConfigLoader
from src.core.utils.sanitize import sanitize_name
from src.practices.article.schema import ArticleFilter
from src.practices.article.dual_validation.validator import DualValidator, compute_metrics

# Ensure prompts (including ob_rqcontext_dual) are registered
import src.practices.article  # noqa: F401

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Article Pairwise (Dual) Validation")
    parser.add_argument("--model", required=True, help="模型名称（对应 model.toml 配置）")
    parser.add_argument("--provider", default=None, help="Provider 名称")
    parser.add_argument("--config_path", default="assets/model.toml", help="配置文件路径")
    parser.add_argument("--prompt", default="ob_rqcontext_dual", help="Prompt 名称")
    parser.add_argument("--entry", default="rq_with_context", help="Article entry 字段")
    parser.add_argument("--subjects", nargs="*", default=None, help="学科过滤列表")
    parser.add_argument("--types", nargs="*", default=None, help="文章类型过滤列表")
    parser.add_argument("--split", default="validate", help="数据集划分 (train/validate/test)")
    parser.add_argument("--data_source", choices=["jsonl", "mongodb"], default="mongodb")
    parser.add_argument("--jsonl_path", default=None, help="JSONL 文件路径（data_source=jsonl 时必填）")
    parser.add_argument("--db_name", default="RQ0", help="MongoDB 数据库名称")
    parser.add_argument("--connection_string", default=None, help="MongoDB 连接字符串")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（default: 42）")
    parser.add_argument(
        "--num_per_pair_type",
        type=int,
        default=50,
        help="每个 tier 配对类型采样数量（4 tier 两两配对共 6 类，default: 50）",
    )
    parser.add_argument("--max_concurrent", type=int, default=20, help="最大并发数")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度")
    parser.add_argument(
        "--thinking_model",
        type=lambda v: v.lower() in ("true", "1", "yes"),
        default=False,
        help="是否为支持 thinking 的模型",
    )
    parser.add_argument(
        "--enable_thinking",
        type=lambda v: v.lower() in ("true", "1", "yes"),
        default=False,
        help="是否启用模型推理输出",
    )
    parser.add_argument("--output_dir", default="./outcome/dual", help="结果输出目录")
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="忽略已有 pair_results.jsonl 缓存，重新推理全部 pair",
    )
    parser.add_argument(
        "--pair_timeout",
        type=float,
        default=300.0,
        metavar="SEC",
        help="单对推理超时（秒），超时后本对下次运行重试；0 表示不超时（默认 300）",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # ---- Model / Provider config ----
    model_config_raw = ConfigLoader.get_model_config(args.config_path, args.model) or {}
    model_config = {
        "model_name": args.model,
        "model_resource": model_config_raw.get("model_resource"),
        "endpoint": model_config_raw.get("endpoint"),
    }
    provider = args.provider or model_config_raw.get("provider")
    if not provider:
        raise ValueError("必须提供 --provider 或在 model.toml 中配置 provider")

    provider_config = ConfigLoader.get_provider_config_with_env_fallback(
        args.config_path, provider
    )
    if not provider_config:
        raise ValueError(f"Provider '{provider}' 配置未找到")

    # ---- Data loader ----
    if args.data_source == "jsonl":
        if not args.jsonl_path:
            raise ValueError("JSONL 模式需要 --jsonl_path")
        from src.core.dataloader.jsonl import JSONLLoader
        loader = JSONLLoader(args.jsonl_path, id_field="title")
        logger.info(f"数据源: JSONL ({args.jsonl_path})")
    else:
        from src.practices.article.models import init_database
        from src.practices.article.loader import ArticleLoader
        conn = args.connection_string or f"mongodb://localhost:27017/{args.db_name}"
        await init_database(conn, db_name=args.db_name)
        loader = ArticleLoader()
        logger.info(f"数据源: MongoDB ({args.db_name})")

    # ---- Build filter & load all articles ----
    article_filter = ArticleFilter(
        split=args.split,
        subjects=args.subjects,
        types=args.types,
    )

    logger.info("Loading articles from MongoDB ...")
    articles = []
    async for item in loader.load_stream(article_filter):
        articles.append(item)
    logger.info(f"Loaded {len(articles)} articles")

    # ---- Validator ----
    safe_model = sanitize_name(args.model)
    validator = DualValidator(
        model_config=model_config,
        provider_config=provider_config,
        entry=args.entry,
        prompt_name=args.prompt,
        max_concurrent=args.max_concurrent,
        temperature=args.temperature,
        num_per_pair_type=args.num_per_pair_type,
        seed=args.seed,
        thinking_model=args.thinking_model,
        enable_thinking=args.enable_thinking,
        pair_timeout=args.pair_timeout if args.pair_timeout > 0 else None,
    )

    # Cache path: reuse pair_results.jsonl from a previous run (same output dir)
    cache_path = os.path.join(args.output_dir, safe_model, "pair_results.jsonl")
    results = await validator.validate(
        articles,
        cache_path=cache_path,
        overwrite_cache=args.overwrite_cache,
    )

    # ---- Metrics ----
    metrics = compute_metrics(results)

    print(f"\n{'='*50}")
    print(f"Model: {args.model}")
    print(f"Total pairs evaluated : {metrics['total']}")
    print(f"Overall accuracy      : {metrics['overall_accuracy']:.2%}")
    print(f"Weighted accuracy     : {metrics['weighted_accuracy']:.2%}")
    if 'alignment_fail_total' in metrics:
        print(f"Alignment fail total  : {metrics['alignment_fail_total']}")
    for pt_key, m in metrics["per_pair_type"].items():
        extra = f"  align_fail={m['alignment_fail']}" if 'alignment_fail' in m else ""
        print(f"  {pt_key}: n={m['n']}  correct={m['correct']}  acc={m['accuracy']:.2%}{extra}")
    print(f"{'='*50}\n")

    # ---- Save ----
    out_dir = os.path.join(args.output_dir, safe_model)
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    results_path = os.path.join(out_dir, "pair_results.jsonl")
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Metrics saved to: {metrics_path}")
    logger.info(f"Pair results saved to: {results_path}")


if __name__ == "__main__":
    asyncio.run(main())
