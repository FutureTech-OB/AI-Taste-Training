"""
统计数据库中每个期刊在 split=train 且 type=study 下的文章数量分布。

用法:
    python -m src.practices.article.scripts.journal_train_study_stats
    python -m src.practices.article.scripts.journal_train_study_stats --db_name RIOB --connection_string "mongodb://..."
"""
import argparse
import asyncio
import sys
from pathlib import Path

# 项目根（scripts -> article -> practices -> src -> RQ）
_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.practices.article.models import Article, init_database


async def main(
    connection_string: str,
    db_name: str,
):
    await init_database(connection_string, db_name=db_name)

    pipeline = [
        {"$match": {"split": "train", "type": "study"}},
        {"$group": {"_id": "$journal", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    rows = await Article.aggregate(pipeline).to_list()

    total = 0
    print("Journal (train + study) 数量分布")
    print("-" * 60)
    for r in rows:
        journal = r["_id"] if r["_id"] is not None else "(无 journal)"
        count = r["count"]
        total += count
        print(f"  {journal[:56]:<58}  {count:>6}")
    print("-" * 60)
    print(f"  {'合计':<58}  {total:>6}")


def _parse_args():
    p = argparse.ArgumentParser(description="每个期刊 train+study 数量分布")
    p.add_argument("--connection_string", default=None, help="MongoDB 连接字符串")
    p.add_argument("--db_name", default="RIOB", help="数据库名")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    conn = args.connection_string or "mongodb://root:password@166.111.96.30:27027/?authSource=admin"
    asyncio.run(main(connection_string=conn, db_name=args.db_name))
