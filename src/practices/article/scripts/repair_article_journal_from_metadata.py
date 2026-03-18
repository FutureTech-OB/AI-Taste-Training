"""
Use metadata.journal.Journal name to repair Article.journal.

Default behavior:
- process ECONOMICS and SOCIOLOGY
- require metadata.journal.Journal name to exist
- update article.journal when it differs from the metadata value
- support dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import unicodedata
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


def _normalize_journal(text: str | None) -> str:
    if not text:
        return ""
    value = unicodedata.normalize("NFKC", str(text)).lower().strip()
    value = value.replace("&", " and ")
    value = re.sub(r"\bthe\b", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


async def run(
    connection_string: str | None = None,
    db_name: str | None = None,
    subjects: Iterable[str] = ("ECONOMICS", "SOCIOLOGY"),
    limit: int | None = None,
    sync_exact: bool = False,
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

    coll = Article.get_pymongo_collection()
    target_subjects = _normalize_subjects(subjects)

    query = {
        "subject": {"$in": target_subjects},
        "metadata.journal.Journal name": {"$exists": True, "$nin": [None, ""]},
    }

    raw_candidates = await coll.find(
        query,
        {
            "_id": 1,
            "subject": 1,
            "journal": 1,
            "metadata.journal.Journal name": 1,
        },
    ).to_list(length=None)

    candidates = []
    for doc in raw_candidates:
        metadata_journal = ((doc.get("metadata") or {}).get("journal") or {}).get("Journal name")
        current_journal = doc.get("journal")
        if sync_exact:
            needs_update = current_journal != metadata_journal
        else:
            needs_update = _normalize_journal(current_journal) != _normalize_journal(metadata_journal)
        if needs_update:
            candidates.append(doc)
            if limit is not None and len(candidates) >= limit:
                break

    print(f"database={db_name}")
    print(f"subjects={','.join(target_subjects)}")
    print(f"sync_exact={sync_exact}")
    print(f"mismatched_candidates={len(candidates)}")

    if not candidates:
        return

    preview = candidates[:10]
    print("preview:")
    for doc in preview:
        metadata_journal = ((doc.get("metadata") or {}).get("journal") or {}).get("Journal name")
        print(
            f"  {doc.get('_id')} | {doc.get('subject')} | "
            f"{doc.get('journal')} -> {metadata_journal}"
        )

    if dry_run:
        print("dry_run=true, no database updates applied")
        return

    updated = 0
    for doc in candidates:
        metadata_journal = ((doc.get("metadata") or {}).get("journal") or {}).get("Journal name")
        result = await coll.update_one(
            {"_id": doc["_id"]},
            {"$set": {"journal": metadata_journal}},
        )
        updated += int(result.modified_count)

    print(f"updated={updated}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair Article.journal using metadata.journal.Journal name"
    )
    parser.add_argument("--connection-string", default=None, help="MongoDB connection string")
    parser.add_argument("--db-name", default=None, help="Database name")
    parser.add_argument("--subjects", nargs="+", default=["ECONOMICS", "SOCIOLOGY"], help="Subjects to process")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of rows to process")
    parser.add_argument("--sync-exact", action="store_true", help="Force article.journal to exactly match metadata text")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run(
            connection_string=args.connection_string,
            db_name=args.db_name,
            subjects=args.subjects,
            limit=args.limit,
            sync_exact=args.sync_exact,
            dry_run=args.dry_run,
        )
    )
