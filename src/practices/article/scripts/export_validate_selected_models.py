"""
Export validate docs from MongoDB, keeping only selected val_outcome models.

Default use case:
- export all validate docs from RItest
- keep only chosen rq_with_context model outputs
- write JSONL for downstream analysis
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
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


def _ordered_doc(doc: dict[str, Any], article_number: int, model_keys: list[str]) -> OrderedDict[str, Any]:
    rq_with_context = (((doc.get("val_outcome") or {}).get("rq_with_context")) or {})
    filtered_rq = OrderedDict()
    for model_key in model_keys:
        if model_key in rq_with_context:
            filtered_rq[model_key] = rq_with_context[model_key]

    ordered = OrderedDict()
    ordered["title"] = doc.get("title")
    ordered["journal"] = doc.get("journal")
    ordered["published_year"] = doc.get("published_year")
    ordered["package"] = 1
    ordered["article_number"] = article_number
    ordered["entries"] = doc.get("entries") or {}
    ordered["rank"] = doc.get("rank")
    ordered["split"] = doc.get("split")
    ordered["subject"] = doc.get("subject")
    ordered["val_outcome"] = {"rq_with_context": filtered_rq}
    return ordered


def export_validate_selected_models(
    out_jsonl: Path,
    model_keys: list[str],
    db_name: str | None = None,
    split: str = "validate",
    subjects: list[str] | None = None,
) -> Path:
    connection_string, resolved_db_name = load_database_config(db_name)
    client = MongoClient(connection_string)
    coll = client[resolved_db_name]["Article"]

    query: dict[str, Any] = {"split": split}
    if subjects:
        query["subject"] = {"$in": subjects}
    total = coll.count_documents(query)
    print(f"database={resolved_db_name}")
    print(f"query={query}")
    print(f"matched={total}")
    print(f"model_keys={len(model_keys)}")

    cursor = coll.find(query).sort([("subject", 1), ("_id", 1)])
    rows: list[str] = []
    kept = 0
    for idx, doc in enumerate(cursor, start=1):
        ordered = _ordered_doc(doc, article_number=idx, model_keys=model_keys)
        rq_models = ((ordered.get("val_outcome") or {}).get("rq_with_context") or {})
        if not rq_models:
            continue
        rows.append(json.dumps(ordered, ensure_ascii=False))
        kept += 1

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"exported={kept}")
    print(f"out_jsonl={out_jsonl}")
    return out_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export validate docs with selected model results")
    parser.add_argument("--out-jsonl", required=True, help="Output JSONL path")
    parser.add_argument("--db-name", default=None, help="Database name")
    parser.add_argument("--split", default="validate", help="Split to export")
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional subject filter")
    parser.add_argument("--models", nargs="+", required=True, help="Model keys to keep")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_validate_selected_models(
        out_jsonl=Path(args.out_jsonl),
        model_keys=args.models,
        db_name=args.db_name,
        split=args.split,
        subjects=args.subjects,
    )
