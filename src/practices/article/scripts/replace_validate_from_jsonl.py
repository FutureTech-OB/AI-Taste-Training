"""
Replace subject validate docs in MongoDB with records from a JSONL file.

Default use case:
- clear current split=validate docs for a subject
- import JSONL rows as new validate docs
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Tuple

from pymongo import MongoClient

try:
    import tomllib
except ImportError:
    import tomli as tomllib


project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def load_database_config(db_name: str | None = None) -> Tuple[str, str]:
    config_path = project_root / "assets" / "database.toml"
    default_db_name = "RQ"
    default_connection_string_tpl = (
        "mongodb://root:password@166.111.96.30:27027/{db}?authSource=admin"
    )

    if not config_path.exists():
        resolved_db_name = db_name or os.getenv("ARTICLE_DB_NAME") or default_db_name
        return default_connection_string_tpl.format(db=resolved_db_name), resolved_db_name

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    default_config = config.get("default", {})
    resolved_db_name = db_name or os.getenv("ARTICLE_DB_NAME") or default_config.get("db_name", default_db_name)
    conn_str = default_config.get("connection_string", "")
    if conn_str:
        connection_string = conn_str.replace("<DBNAME>", resolved_db_name) if "<DBNAME>" in conn_str else conn_str
    else:
        connection_string = default_connection_string_tpl.format(db=resolved_db_name)
    return connection_string, resolved_db_name


def _build_article_doc(row: dict[str, Any], source_jsonl: str) -> dict[str, Any]:
    now = datetime.datetime.now()
    metadata = {
        "import_source": source_jsonl,
    }
    if row.get("domain") is not None:
        metadata["domain"] = row.get("domain")
    if row.get("level") is not None:
        metadata["level"] = row.get("level")
    if row.get("package") is not None:
        metadata["package"] = row.get("package")
    if row.get("article_number") is not None:
        metadata["article_number"] = row.get("article_number")

    return {
        "title": row.get("title"),
        "published_year": row.get("published_year"),
        "journal": row.get("journal"),
        "type": row.get("type") or "study",
        "rank": row.get("rank"),
        "split": "validate",
        "subject": row.get("subject"),
        "metadata": metadata,
        "entries": row.get("entries") or {},
        "status": "entries_parsed",
        "trial_count": 0,
        "abort": False,
        "created_at": now,
        "updated_at": now,
    }


def replace_validate_from_jsonl(
    jsonl_path: Path,
    subject: str,
    db_name: str | None = None,
    dry_run: bool = False,
) -> None:
    connection_string, resolved_db_name = load_database_config(db_name)
    client = MongoClient(connection_string)
    coll = client[resolved_db_name]["Article"]

    normalized_subject = subject.strip()
    existing_filter = {"subject": normalized_subject, "split": "validate"}

    with jsonl_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    imported_docs = []
    for row in rows:
        row_subject = (row.get("subject") or "").strip()
        if row_subject and row_subject != normalized_subject:
            raise ValueError(f"JSONL subject mismatch: expected {normalized_subject}, got {row_subject}")
        row["subject"] = normalized_subject
        imported_docs.append(_build_article_doc(row, str(jsonl_path)))

    existing_count = coll.count_documents(existing_filter)
    print(f"database={resolved_db_name}")
    print(f"subject={normalized_subject}")
    print(f"jsonl_rows={len(imported_docs)}")
    print(f"existing_validate={existing_count}")
    print(f"dry_run={dry_run}")

    if dry_run:
        return

    if existing_count:
        coll.delete_many(existing_filter)
    if imported_docs:
        coll.insert_many(imported_docs, ordered=True)

    final_count = coll.count_documents(existing_filter)
    print(f"final_validate={final_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replace subject validate docs from a JSONL file")
    parser.add_argument("--jsonl", required=True, help="Source JSONL path")
    parser.add_argument("--subject", required=True, help="Subject to replace")
    parser.add_argument("--db-name", default=None, help="Database name")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    replace_validate_from_jsonl(
        jsonl_path=Path(args.jsonl),
        subject=args.subject,
        db_name=args.db_name,
        dry_run=args.dry_run,
    )
