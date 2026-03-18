"""
Export validate-set Article docs for a specified subject into JSONL.

Default filters:
- subject = required
- split = validate
- status = entries_parsed
- type = study

Optional filters:
- entries.<entry> contains non-whitespace text
- published_year
- limit
- val_outcome sidecar export

Examples:
  python -m src.practices.article.scripts.export_validate_by_subject --subject ECONOMICS

  $env:ARTICLE_DB_NAME="RItest"
  python -m src.practices.article.scripts.export_validate_by_subject \
    --subject SOCIOLOGY \
    --entry rq_with_context \
    --out-jsonl exports/sociology_validate.jsonl \
    --valoutput exports/sociology_validate_val_outcome.jsonl
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


def build_query(
    subject: str,
    split: str,
    status: str,
    article_type: str,
    entry: str | None,
    year: int | None,
) -> dict[str, Any]:
    query: dict[str, Any] = {
        "subject": subject.strip().upper(),
        "split": split,
        "status": status,
        "type": article_type,
    }
    if entry:
        query[f"entries.{entry}"] = {"$regex": r"\S"}
    if year is not None:
        query["published_year"] = int(year)
    return query


def _build_valoutput_doc(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "_id": doc.get("_id"),
        "title": doc.get("title"),
        "subject": doc.get("subject"),
        "split": doc.get("split"),
        "status": doc.get("status"),
        "type": doc.get("type"),
        "published_year": doc.get("published_year"),
        "journal": doc.get("journal"),
        "rank": doc.get("rank"),
        "entries": doc.get("entries"),
        "val_outcome": doc.get("val_outcome"),
    }


def export_validate_by_subject(
    subject: str,
    out_jsonl: Path | None = None,
    valoutput: Path | None = None,
    db_name: str | None = None,
    split: str = "validate",
    status: str = "entries_parsed",
    article_type: str = "study",
    entry: str | None = "rq_with_context",
    year: int | None = None,
    limit: int | None = None,
) -> Path:
    connection_string, resolved_db_name = load_database_config(db_name)
    client = MongoClient(connection_string)
    coll = client[resolved_db_name]["Article"]

    normalized_subject = subject.strip().upper()
    query = build_query(
        subject=normalized_subject,
        split=split,
        status=status,
        article_type=article_type,
        entry=entry,
        year=year,
    )

    if out_jsonl is None:
        filename = f"{resolved_db_name.lower()}_{normalized_subject.lower()}_{split}.jsonl"
        out_jsonl = project_root / "exports" / filename

    total = coll.count_documents(query)
    print(f"database={resolved_db_name}")
    print(f"query={query}")
    print(f"matched={total}")

    cursor = coll.find(query).sort([("published_year", 1), ("_id", 1)])
    if limit is not None:
        cursor = cursor.limit(limit)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if valoutput is not None:
        valoutput.parent.mkdir(parents=True, exist_ok=True)
    exported = 0
    valoutput_exported = 0
    with out_jsonl.open("w", encoding="utf-8") as f:
        val_f = valoutput.open("w", encoding="utf-8") if valoutput is not None else None
        try:
            for doc in cursor:
                doc = _sanitize(doc)
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                exported += 1

                if val_f is not None:
                    val_doc = _build_valoutput_doc(doc)
                    val_f.write(json.dumps(val_doc, ensure_ascii=False) + "\n")
                    valoutput_exported += 1
        finally:
            if val_f is not None:
                val_f.close()

    print(f"exported={exported}")
    print(f"out_jsonl={out_jsonl}")
    if valoutput is not None:
        print(f"valoutput_exported={valoutput_exported}")
        print(f"valoutput={valoutput}")
    return out_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export subject-specific validate Article docs to JSONL"
    )
    parser.add_argument("--subject", required=True, help="Subject name, e.g. ECONOMICS")
    parser.add_argument("--db-name", default=None, help="Database name; defaults to ARTICLE_DB_NAME or database.toml")
    parser.add_argument("--out-jsonl", default=None, help="Output JSONL path")
    parser.add_argument("--valoutput", default=None, help="Optional val_outcome sidecar JSONL path")
    parser.add_argument("--split", default="validate", help="Dataset split to export")
    parser.add_argument("--status", default="entries_parsed", help="Required status")
    parser.add_argument("--type", dest="article_type", default="study", help="Required article type")
    parser.add_argument("--entry", default="rq_with_context", help="Require non-empty entries.<entry>; set empty string to disable")
    parser.add_argument("--year", type=int, default=None, help="Optional published_year filter")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of documents")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_validate_by_subject(
        subject=args.subject,
        out_jsonl=Path(args.out_jsonl) if args.out_jsonl else None,
        valoutput=Path(args.valoutput) if args.valoutput else None,
        db_name=args.db_name,
        split=args.split,
        status=args.status,
        article_type=args.article_type,
        entry=(args.entry or None),
        year=args.year,
        limit=args.limit,
    )
