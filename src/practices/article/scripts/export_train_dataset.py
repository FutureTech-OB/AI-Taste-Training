"""
Export train dataset into a JSON array file for downstream use.

Default source:
- ARTICLE_DB_NAME or database.toml

Default filter:
- split = train

Output format matches reports/data/train_data style and keeps all entries.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Tuple

from bson import DBRef, ObjectId
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


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, DBRef):
        return {"$ref": value.collection, "$id": str(value.id), "$db": value.database}
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    return value


def _project_doc(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": doc.get("title"),
        "journal": doc.get("journal"),
        "published_year": doc.get("published_year"),
        "rank": doc.get("rank"),
        "type": doc.get("type"),
        "split": doc.get("split"),
        "subject": doc.get("subject"),
        "entries": _sanitize(doc.get("entries") or {}),
    }


def export_train_dataset(
    out_json: Path,
    db_name: str | None = None,
    split: str = "train",
    types: str | None = "study",
    status: str | None = "entries_parsed",
) -> Path:
    connection_string, resolved_db_name = load_database_config(db_name)
    client = MongoClient(connection_string)
    coll = client[resolved_db_name]["Article"]

    query: dict[str, Any] = {"split": split}
    if types:
        query["type"] = types
    if status:
        query["status"] = status

    total = coll.count_documents(query)
    print(f"database={resolved_db_name}")
    print(f"query={query}")
    print(f"matched={total}")

    cursor = coll.find(query).sort([("subject", 1), ("published_year", 1), ("_id", 1)])
    rows = [_project_doc(doc) for doc in cursor]

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"exported={len(rows)}")
    print(f"out_json={out_json}")
    return out_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export train dataset JSON with all entries")
    parser.add_argument("--db-name", default=None, help="Database name; defaults to ARTICLE_DB_NAME or database.toml")
    parser.add_argument("--out-json", default=None, help="Output JSON path")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--type", dest="article_type", default="study", help="Article type; set empty string to disable")
    parser.add_argument("--status", default="entries_parsed", help="Status; set empty string to disable")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_json = (
        Path(args.out_json)
        if args.out_json
        else project_root / "reports" / "data" / "train_data" / f"{(args.db_name or os.getenv('ARTICLE_DB_NAME') or 'RQ')}.Article.train.json"
    )
    export_train_dataset(
        out_json=out_json,
        db_name=args.db_name,
        split=args.split,
        types=(args.article_type or None),
        status=(args.status or None),
    )
