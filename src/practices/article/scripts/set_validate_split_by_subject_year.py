"""
按学科和 rank 为指定年份设置 validate 集，并将其余目标文档设为 train。

默认规则：
- 只处理 status=entries_parsed
- 学科默认 ECONOMICS、SOCIOLOGY
- validate 候选只来自 published_year=2025
- 每个学科每个 rank 默认 50 篇 validate
- rank 默认 Exceptional、Strong、Fair、Limited
- 先检查当前各 rank 的 validate 数量；不足时只补差值
- 同范围内其他非 validate 文档统一设为 train
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
from pathlib import Path
from typing import Iterable, Tuple

from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

try:
    import tomllib
except ImportError:
    import tomli as tomllib


project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.practices.article.models import Article


DEFAULT_RANKS = ["exceptional", "strong", "fair", "limited"]


def load_database_config() -> Tuple[str, str]:
    config_path = project_root / "assets" / "database.toml"
    default_db_name = "RQ"
    default_connection_string_tpl = (
        "mongodb://root:password@166.111.96.30:27027/{db}?authSource=admin"
    )

    if not config_path.exists():
        return default_connection_string_tpl.format(db=default_db_name), default_db_name

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    default_config = config.get("default", {})
    db_name = os.getenv("ARTICLE_DB_NAME") or default_config.get("db_name", default_db_name)
    conn_str = default_config.get("connection_string", "")
    if conn_str:
        connection_string = conn_str.replace("<DBNAME>", db_name) if "<DBNAME>" in conn_str else conn_str
    else:
        connection_string = default_connection_string_tpl.format(db=db_name)
    return connection_string, db_name


def _normalize_subjects(subjects: Iterable[str]) -> list[str]:
    result = []
    for item in subjects:
        value = str(item or "").strip().upper()
        if value:
            result.append(value)
    return result


def _normalize_ranks(ranks: Iterable[str]) -> list[str]:
    result = []
    for item in ranks:
        value = str(item or "").strip()
        if not value:
            continue
        result.append(value.lower())
    return result


async def run(
    connection_string: str | None = None,
    db_name: str | None = None,
    subjects: Iterable[str] = ("ECONOMICS", "SOCIOLOGY"),
    year: int = 2025,
    validate_per_rank: int = 50,
    ranks: Iterable[str] = DEFAULT_RANKS,
    status: str = "entries_parsed",
    seed: int = 42,
    dry_run: bool = False,
) -> None:
    if connection_string is None or db_name is None:
        conn_str, name = load_database_config()
        connection_string = connection_string or conn_str
        db_name = db_name or name

    if "<DBNAME>" in connection_string:
        connection_string = connection_string.replace("<DBNAME>", db_name)

    client = AsyncIOMotorClient(connection_string)
    database = client[db_name]
    await init_beanie(database=database, document_models=[Article])

    coll = Article.get_motor_collection()
    target_subjects = _normalize_subjects(subjects)
    target_ranks = _normalize_ranks(ranks)

    print(f"database={db_name}")
    print(
        f"subjects={','.join(target_subjects)} year={year} "
        f"status={status} ranks={','.join(target_ranks)} validate_per_rank={validate_per_rank}"
    )

    for subject in target_subjects:
        subject_scope_filter = {
            "subject": subject,
            "status": status,
        }
        year_scope_filter = {
            **subject_scope_filter,
            "published_year": year,
        }

        if dry_run:
            reset_count = await coll.count_documents(
                {
                    **subject_scope_filter,
                    "$or": [
                        {"split": {"$exists": False}},
                        {"split": None},
                        {"split": {"$ne": "validate"}},
                    ],
                }
            )
        else:
            reset_result = await coll.update_many(
                {
                    **subject_scope_filter,
                    "$or": [
                        {"split": {"$exists": False}},
                        {"split": None},
                        {"split": {"$ne": "validate"}},
                    ],
                },
                {"$set": {"split": "train"}},
            )
            reset_count = int(reset_result.modified_count)

        print(f"[{subject}] reset_non_validate_to_train={reset_count}")

        total_validate_after_topup = 0
        for rank in target_ranks:
            rank_label = rank.capitalize()
            rank_scope_filter = {
                **year_scope_filter,
                "rank": rank,
            }
            current_validate_filter = {
                **rank_scope_filter,
                "split": "validate",
            }
            current_validate_count = await coll.count_documents(current_validate_filter)
            total_rank_candidates = await coll.count_documents(rank_scope_filter)

            print(
                f"[{subject}][{rank_label}] year_candidates={total_rank_candidates} "
                f"current_validate={current_validate_count}"
            )

            if total_rank_candidates < validate_per_rank:
                raise ValueError(
                    f"{subject} / {rank_label} 在 {year} 年只有 {total_rank_candidates} 篇 {status}，不足 {validate_per_rank} 篇"
                )

            if current_validate_count >= validate_per_rank:
                total_validate_after_topup += current_validate_count
                print(f"[{subject}][{rank_label}] 已满足，不补")
                continue

            gap = validate_per_rank - current_validate_count
            top_up_filter = {
                **rank_scope_filter,
                "$or": [
                    {"split": {"$exists": False}},
                    {"split": None},
                    {"split": {"$ne": "validate"}},
                ],
            }
            candidates = await coll.find(top_up_filter, {"_id": 1}).to_list(length=None)
            ordered_ids = sorted(str(doc["_id"]) for doc in candidates)

            if len(ordered_ids) < gap:
                raise ValueError(
                    f"{subject} / {rank_label} 可补候选仅 {len(ordered_ids)} 篇，不足补齐 {gap} 篇"
                )

            rng = random.Random(f"{seed}:{subject}:{rank}:{year}:{validate_per_rank}")
            selected_ids = set(rng.sample(ordered_ids, gap))
            selected_object_ids = [doc["_id"] for doc in candidates if str(doc["_id"]) in selected_ids]

            if dry_run:
                total_validate_after_topup += current_validate_count + gap
                print(f"[{subject}][{rank_label}] 需补 {gap} 篇 validate；dry_run 未写库")
                continue

            update_result = await coll.update_many(
                {"_id": {"$in": selected_object_ids}},
                {"$set": {"split": "validate"}},
            )
            total_validate_after_topup += current_validate_count + int(update_result.modified_count)
            print(f"[{subject}][{rank_label}] 补齐 {gap} 篇，实际更新 {update_result.modified_count} 篇")

        print(f"[{subject}] validate_total_after_check={total_validate_after_topup}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按学科和 rank 为指定年份补齐 validate 集，并将其余 entries_parsed 文档设为 train"
    )
    parser.add_argument("--connection-string", default=None, help="MongoDB 连接字符串")
    parser.add_argument("--db-name", default=None, help="数据库名")
    parser.add_argument("--subjects", nargs="+", default=["ECONOMICS", "SOCIOLOGY"], help="学科列表")
    parser.add_argument("--year", type=int, default=2025, help="validate 候选年份")
    parser.add_argument("--validate-per-rank", type=int, default=50, help="每个 rank 的 validate 数量")
    parser.add_argument("--ranks", nargs="+", default=DEFAULT_RANKS, help="rank 列表")
    parser.add_argument("--status", default="entries_parsed", help="限定状态")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不写数据库")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run(
            connection_string=args.connection_string,
            db_name=args.db_name,
            subjects=args.subjects,
            year=args.year,
            validate_per_rank=args.validate_per_rank,
            ranks=args.ranks,
            status=args.status,
            seed=args.seed,
            dry_run=args.dry_run,
        )
    )
