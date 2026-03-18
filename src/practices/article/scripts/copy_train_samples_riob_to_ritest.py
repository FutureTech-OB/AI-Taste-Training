"""
Copy selected Article samples from a source DB to a target DB.

Default source:
- RIOB

Default target:
- RItest

Default filter:
{
  "entries.rq_with_context": {"$ne": None},
  "type": "study",
  "split": "train"
}

On copied target documents:
- set status = "entries_parsed"
- preserve the same _id
- upsert into target
- copy linked PDFData when present
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Tuple

from bson import DBRef
from pymongo import MongoClient, ReplaceOne

try:
    import tomllib
except ImportError:
    import tomli as tomllib


project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def load_database_config(db_name: str) -> Tuple[str, str]:
    config_path = project_root / "assets" / "database.toml"
    default_connection_string_tpl = (
        "mongodb://root:password@166.111.96.30:27027/{db}?authSource=admin"
    )

    if not config_path.exists():
        return default_connection_string_tpl.format(db=db_name), db_name

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    default_config = config.get("default", {})
    conn_str = default_config.get("connection_string", "")
    if conn_str:
        connection_string = conn_str.replace("<DBNAME>", db_name) if "<DBNAME>" in conn_str else conn_str
    else:
        connection_string = default_connection_string_tpl.format(db=db_name)
    return connection_string, db_name


def _extract_pdfdata_id(value: Any):
    if value is None:
        return None
    if isinstance(value, DBRef):
        return value.id
    if isinstance(value, dict):
        if "$id" in value:
            return value["$id"]
        if "id" in value:
            return value["id"]
    return None


def run(
    source_db: str = "RIOB",
    target_db: str = "RItest",
    limit: int | None = None,
    dry_run: bool = False,
) -> None:
    source_conn, _ = load_database_config(source_db)
    target_conn, _ = load_database_config(target_db)

    source_client = MongoClient(source_conn)
    target_client = MongoClient(target_conn)

    source_article = source_client[source_db]["Article"]
    source_pdf = source_client[source_db]["PDFData"]
    target_article = target_client[target_db]["Article"]
    target_pdf = target_client[target_db]["PDFData"]

    query = {
        "entries.rq_with_context": {"$ne": None},
        "type": "study",
        "split": "train",
    }

    total = source_article.count_documents(query)
    print(f"source_db={source_db}")
    print(f"target_db={target_db}")
    print(f"matched_articles={total}")
    print(f"dry_run={dry_run}")

    cursor = source_article.find(query)
    if limit is not None:
        cursor = cursor.limit(limit)

    article_ops = []
    pdf_ops = []
    seen_pdf_ids = set()
    copied = 0
    linked_pdf_count = 0

    for doc in cursor:
        copied += 1
        doc["status"] = "entries_parsed"

        pdf_id = _extract_pdfdata_id(doc.get("pdfdata"))
        if pdf_id is not None and pdf_id not in seen_pdf_ids:
            pdf_doc = source_pdf.find_one({"_id": pdf_id})
            if pdf_doc is not None:
                pdf_ops.append(ReplaceOne({"_id": pdf_id}, pdf_doc, upsert=True))
                seen_pdf_ids.add(pdf_id)
                linked_pdf_count += 1

        article_ops.append(ReplaceOne({"_id": doc["_id"]}, doc, upsert=True))

    print(f"to_copy_articles={copied}")
    print(f"to_copy_pdfdata={linked_pdf_count}")

    if not dry_run and pdf_ops:
        result = target_pdf.bulk_write(pdf_ops, ordered=False)
        print(
            "pdfdata_upserted="
            f"{result.upserted_count + result.modified_count}"
        )

    if not dry_run and article_ops:
        result = target_article.bulk_write(article_ops, ordered=False)
        print(
            "article_upserted="
            f"{result.upserted_count + result.modified_count}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy train study samples with rq_with_context from RIOB to RItest"
    )
    parser.add_argument("--source-db", default="RIOB", help="Source database name")
    parser.add_argument("--target-db", default="RItest", help="Target database name")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of articles")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        source_db=args.source_db,
        target_db=args.target_db,
        limit=args.limit,
        dry_run=args.dry_run,
    )
