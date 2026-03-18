"""
将指定期刊下、满足训练条件的文章，有一半标记为 split=abandon。

默认训练条件与 SFT 一致：
- subject=ob
- split=train（作为“来源 split”）
- type=study
- entry=rq_with_context 非空（含非空白字符）

示例：
  python -m src.practices.article.scripts.abandon_half_by_journal \
    --journal "Strategic Management Journal" \
    --subject ob --from_split train --types study --entry rq_with_context \
    --seed 42 --fraction 0.5 --dry_run

实际执行（更新数据库）时去掉 --dry_run。
"""
import argparse
import asyncio
import math
import random
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import toml
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from bson import ObjectId

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


def _build_match(journal: str, subject: Optional[str], from_split: Optional[str], types: Optional[str], entry: str) -> Dict[str, Any]:
    match: Dict[str, Any] = {
        "journal": journal,
        # entry 非空（含非空白字符）
        f"entries.{entry}": {"$regex": "\\S"},
    }
    if subject:
        match["subject"] = subject
    if from_split:
        match["split"] = from_split
    if types:
        match["type"] = types
    return match


async def main():
    parser = argparse.ArgumentParser(description="将指定期刊下的一半文章标记为 split=abandon（满足训练条件）")
    parser.add_argument("--journal", required=True, help="期刊名，例如 'Strategic Management Journal'")
    parser.add_argument("--subject", default="ob")
    parser.add_argument("--from_split", default="train", help="来源 split（仅从该 split 中抽样）")
    parser.add_argument("--types", default="study")
    parser.add_argument("--entry", default="rq_with_context")
    parser.add_argument("--fraction", type=float, default=0.5, help="抽样比例，默认一半")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="仅显示将要更新的数量，不写入")
    parser.add_argument("--connection_string", default=None)
    parser.add_argument("--db_name", default=None)
    args = parser.parse_args()

    if not (0 < args.fraction <= 1):
        raise ValueError("--fraction 必须在 (0, 1] 区间内")

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

    match = _build_match(args.journal, args.subject, args.from_split, args.types, args.entry)

    # 仅取 _id 列表以便随机抽样
    coll = Article.get_motor_collection()
    cursor = coll.find(match, {"_id": 1})
    ids: List[ObjectId] = [doc["_id"] async for doc in cursor]

    total = len(ids)
    target = math.floor(total * args.fraction)

    print("筛选条件：")
    print(f"  journal={args.journal}")
    print(f"  subject={args.subject} from_split={args.from_split} type={args.types} entry={args.entry}")
    print(f"  db={db}\n  conn={conn}")
    print(f"匹配到文章总数：{total}")
    print(f"计划标记为 abandon 的数量：{target} （比例={args.fraction}，seed={args.seed}）")

    if target <= 0:
        print("无需更新（目标数量为 0）")
        return

    rnd = random.Random(args.seed)
    selected = set(rnd.sample(ids, target))

    if args.dry_run:
        print("dry_run 模式：不执行数据库写入。")
        return

    # 执行批量更新
    result = await coll.update_many({"_id": {"$in": list(selected)}}, {"$set": {"split": "abandon"}})
    print(f"已更新文档数：{getattr(result, 'modified_count', None) or getattr(result, 'raw_result', {}).get('nModified')}")


if __name__ == "__main__":
    asyncio.run(main())

