"""按期刊名称回填 Article.rank。"""

from __future__ import annotations

import argparse
import asyncio
import re
import unicodedata
from pathlib import Path
from collections import Counter
from typing import Dict, Optional, Tuple

try:
    import tomllib
except ImportError:
    import tomli as tomllib

import sys

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
src_root = project_root / "src"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from src.practices.article.models import Article, init_database


def _normalize_journal_name(name: Optional[str]) -> str:
    if not name:
        return ""
    text = unicodedata.normalize("NFKC", str(name)).lower().strip()
    text = text.replace("&", " and ")
    text = text.replace("?", " ")
    text = re.sub(r"\bthe\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_rank_mapping() -> Dict[str, str]:
    tiers = {
        "exceptional": [
            "Academy of Management Journal",
            "Administrative Science Quarterly",
            "Academy of Management Review",
            "Organization Science",
            "Organisation Science",
            "Strategic Management Journal",
            "Journal of Applied Psychology",
        ],
        "strong": [
            "Organizational Behavior and Human Decision Processes",
            "Organisational Behavior and Human Decision Processes",
            "Organizational Behaviour and Human Decision Processes",
            "Organisational Behaviour and Human Decision Processes",
            "Journal of Management",
            "Personnel Psychology",
        ],
        "fair": [
            "Journal of Management Studies",
            "Journal of Organizational Behavior",
            "Journal of Organisational Behavior",
            "Journal of Organizational Behaviour",
            "Journal of Organisational Behaviour",
            "Leadership Quarterly",
            "The Leadership Quarterly",
            "Leadership Quarterly (或 The Leadership Quarterly)",
            "Human Resource Management",
            "Human Relations",
        ],
        "limited": [
            "Group and Organization Management",
            "Group and Organisation Management",
            "Journal of Business and Psychology",
            "Journal of Managerial Psychology",
            "Journal of Personnel Psychology",
            "Journal of Organizational Behavior Management",
            "Journal of Organisational Behavior Management",
            "Journal of Organizational Behaviour Management",
            "Journal of Organisational Behaviour Management",
        ],
    }

    mapping: Dict[str, str] = {}
    for rank, journals in tiers.items():
        for journal in journals:
            mapping[_normalize_journal_name(journal)] = rank
    return mapping


def _load_database_config() -> Tuple[str, str]:
    config_path = project_root / "assets" / "database.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    default_config = config.get("default", {})
    db_name = default_config.get("db_name", "RQ")
    conn_str = default_config.get("connection_string", "")
    if "<DBNAME>" in conn_str:
        conn_str = conn_str.replace("<DBNAME>", db_name)
    if not conn_str:
        raise ValueError("无法从 assets/database.toml 读取 connection_string")
    return conn_str, db_name


async def fill_rank_by_journal(
    connection_string: Optional[str] = None,
    db_name: Optional[str] = None,
    overwrite: bool = False,
    set_unmatched_others: bool = False,
    only_study: bool = True,
    dry_run: bool = False,
) -> None:
    cfg_conn, cfg_db = _load_database_config()
    connection_string = connection_string or cfg_conn
    db_name = db_name or cfg_db

    if "<DBNAME>" in connection_string:
        connection_string = connection_string.replace("<DBNAME>", db_name)

    await init_database(connection_string=connection_string)
    mapping = _build_rank_mapping()

    matched_updates = 0
    unmatched_updates = 0
    skipped_existing = 0
    no_journal = 0
    unmatched_count = 0
    total = 0
    unmatched_journals: Counter[str] = Counter()

    base_filter = {"type": "study"} if only_study else {}

    async for article in Article.find(base_filter):
        total += 1
        current_rank = (article.rank.value if hasattr(article.rank, "value") else article.rank) if article.rank else None
        if current_rank and not overwrite:
            skipped_existing += 1
            continue

        normalized_journal = _normalize_journal_name(article.journal)
        if not normalized_journal:
            no_journal += 1
            continue

        target_rank = mapping.get(normalized_journal)
        if target_rank is None:
            unmatched_count += 1
            unmatched_journals[article.journal or "<empty>"] += 1
            if set_unmatched_others:
                target_rank = "others"
            else:
                continue

        if current_rank == target_rank:
            continue

        if not dry_run:
            article.rank = target_rank
            await article.save()

        if target_rank == "others":
            unmatched_updates += 1
        else:
            matched_updates += 1

    print("=" * 72)
    print(f"DB: {db_name}")
    print(f"Total scanned: {total}")
    print(f"Updated (tier matched): {matched_updates}")
    print(f"Updated (set others): {unmatched_updates}")
    print(f"Skipped (existing rank): {skipped_existing}")
    print(f"Skipped (missing journal): {no_journal}")
    print(f"Unmatched journals: {unmatched_count}")
    print(f"Only study: {only_study}")
    print(f"Dry run: {dry_run}")
    if unmatched_journals:
        print("Top unmatched journals:")
        for journal, cnt in unmatched_journals.most_common(20):
            print(f"  - {journal}: {cnt}")
    print("=" * 72)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按期刊分层回填 Article.rank")
    parser.add_argument("--connection-string", default=None, help="MongoDB 连接字符串")
    parser.add_argument("--db-name", default=None, help="数据库名")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有 rank")
    parser.add_argument(
        "--set-unmatched-others",
        action="store_true",
        help="未匹配期刊设置为 others（默认不设置）",
    )
    parser.add_argument(
        "--all-types",
        action="store_true",
        help="处理所有 type（默认仅处理 type=study）",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅统计，不写入数据库")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(
        fill_rank_by_journal(
            connection_string=args.connection_string,
            db_name=args.db_name,
            overwrite=args.overwrite,
            set_unmatched_others=args.set_unmatched_others,
            only_study=not args.all_types,
            dry_run=args.dry_run,
        )
    )
