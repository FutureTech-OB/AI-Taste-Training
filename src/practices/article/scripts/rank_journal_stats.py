"""
统计：每个 rank 符合训练条件的期刊数量

训练条件与 scripts/sft/openai_sft.sh 一致：
- db: assets/database.toml -> default.RIOB（可覆盖）
- split = train
- subject = ob（可覆盖）
- type = study（可覆盖）
- entry = rq_with_context 非空（可覆盖）
- rank in [exceptional, strong, fair, limited]

用法示例：
  python -m src.practices.article.scripts.rank_journal_stats \
    --subject ob \
    --split train \
    --types study \
    --entry rq_with_context

可选：指定连接串/库名（否则读取 assets/database.toml）
  --connection_string "mongodb://.../RIOB?authSource=admin" --db_name RIOB
"""
import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import toml
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

from src.practices.article.models import Article


def _load_db_config() -> Tuple[str, str]:
    assets = Path("assets")
    cfg_path = assets / "database.toml"
    default_db = "RIOB"
    default_conn = f"mongodb://root:password@166.111.96.30:27027/{default_db}?authSource=admin"
    if not cfg_path.exists():
        return default_conn, default_db
    try:
        data = toml.load(cfg_path)
        default = data.get("default", {})
        db_name = default.get("db_name", default_db)
        conn = default.get("connection_string") or default_conn.replace(default_db, db_name)
        if "<DBNAME>" in conn:
            conn = conn.replace("<DBNAME>", db_name)
        return conn, db_name
    except Exception:
        return default_conn, default_db


def _build_pipeline(subject: str, split: str, types: Optional[str], entry: str) -> list[Dict[str, Any]]:
    # 训练条件匹配
    match: Dict[str, Any] = {
        "subject": subject,
        "split": split,
        # rank 仅统计四个有效类别
        "rank": {"$in": ["exceptional", "strong", "fair", "limited"]},
        # 期刊非空
        "journal": {"$regex": "\\S"},
        # entry 非空（包含至少一个非空白字符）
        f"entries.{entry}": {"$regex": "\\S"},
    }
    if types:
        match["type"] = types

    pipeline = [
        {"$match": match},
        # 按 (rank, journal) 统计文章数量
        {"$group": {
            "_id": {"rank": "$rank", "journal": "$journal"},
            "article_count": {"$sum": 1}
        }},
        # 排序：rank -> 文章数 desc -> 期刊名 asc
        {"$sort": {"_id.rank": 1, "article_count": -1, "_id.journal": 1}},
    ]
    return pipeline


async def main():
    parser = argparse.ArgumentParser(description="统计每个 rank 符合训练条件的期刊数量")
    parser.add_argument("--subject", default="ob")
    parser.add_argument("--split", default="train")
    parser.add_argument("--types", default="study")
    parser.add_argument("--entry", default="rq_with_context")
    parser.add_argument("--connection_string", default=None)
    parser.add_argument("--db_name", default=None)
    args = parser.parse_args()

    conn, db = args.connection_string, args.db_name
    if not conn or not db:
        cfg_conn, cfg_db = _load_db_config()
        conn = conn or cfg_conn
        db = db or cfg_db
    if "<DBNAME>" in conn:
        conn = conn.replace("<DBNAME>", db)

    client = AsyncIOMotorClient(conn)
    database = client[db]
    await init_beanie(database=database, document_models=[Article])

    pipeline = _build_pipeline(args.subject, args.split, args.types, args.entry)
    results = await Article.aggregate(pipeline).to_list()

    # 固定 rank 顺序
    order = ["exceptional", "strong", "fair", "limited"]
    grouped: Dict[str, list[tuple[str, int]]] = {r: [] for r in order}
    totals: Dict[str, int] = {r: 0 for r in order}
    for doc in results:
        rid = doc.get("_id", {})
        rank = rid.get("rank")
        journal = rid.get("journal") or ""
        cnt = int(doc.get("article_count", 0))
        if rank in grouped:
            grouped[rank].append((journal, cnt))
            totals[rank] += cnt

    print()
    print("统计条件:")
    print(f"  subject={args.subject} split={args.split} type={args.types} entry={args.entry}")
    print(f"  db={db}\n  conn={conn}")
    print()
    print("每个 rank 下各期刊的文章数量（符合训练条件）:")
    for rk in order:
        items = grouped[rk]
        if not items:
            print(f"- {rk}: 无记录")
            continue
        print(f"- {rk} (期刊数={len(items)}, 文章总数={totals[rk]}):")
        for journal, cnt in items:
            print(f"    {journal}: {cnt}")


if __name__ == "__main__":
    asyncio.run(main())
