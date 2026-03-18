"""
Remove specific model keys from Article.val_outcome in MongoDB.

Targets any val_outcome[entry][model_key] and optionally cleans empty entry maps.
"""

from __future__ import annotations

import argparse
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


def remove_model_keys(
    model_keys: list[str],
    db_name: str | None = None,
    dry_run: bool = False,
) -> None:
    connection_string, resolved_db_name = load_database_config(db_name)
    client = MongoClient(connection_string)
    coll = client[resolved_db_name]["Article"]

    print(f"database={resolved_db_name}")
    print(f"model_keys={len(model_keys)}")
    print(f"dry_run={dry_run}")

    docs_scanned = 0
    docs_changed = 0
    removals = 0

    cursor = coll.find({"val_outcome": {"$exists": True}}, {"val_outcome": 1})
    for doc in cursor:
        docs_scanned += 1
        val_outcome = doc.get("val_outcome")
        if not isinstance(val_outcome, dict):
            continue

        changed = False
        for entry_name, entry_map in list(val_outcome.items()):
            if not isinstance(entry_map, dict):
                continue
            for model_key in model_keys:
                if model_key in entry_map:
                    entry_map.pop(model_key, None)
                    removals += 1
                    changed = True
            if not entry_map:
                val_outcome.pop(entry_name, None)

        update_doc: dict[str, Any] | None = None
        if not val_outcome:
            update_doc = {"$unset": {"val_outcome": ""}}
        elif changed:
            update_doc = {"$set": {"val_outcome": val_outcome}}

        if update_doc is None:
            continue

        docs_changed += 1
        if not dry_run:
            coll.update_one({"_id": doc["_id"]}, update_doc)

    print(f"docs_scanned={docs_scanned}")
    print(f"docs_changed={docs_changed}")
    print(f"removals={removals}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove model keys from MongoDB val_outcome")
    parser.add_argument("model_keys", nargs="+", help="Model keys to remove")
    parser.add_argument("--db-name", default=None, help="Database name")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    remove_model_keys(
        model_keys=args.model_keys,
        db_name=args.db_name,
        dry_run=args.dry_run,
    )
