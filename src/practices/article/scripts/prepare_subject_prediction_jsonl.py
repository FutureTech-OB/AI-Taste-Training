"""
Prepare subject validate JSONL files for model prediction analysis.

Source files:
- exports/economics_validate_val_outcome.jsonl
- exports/sociology_validate_val_outcome.jsonl

Output files:
- reports/data/model_predictions/subjects/200_econ.jsonl
- reports/data/model_predictions/subjects/200_social.jsonl
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


project_root = Path(__file__).resolve().parent.parent.parent.parent.parent


def _ordered_doc(doc: dict[str, Any], article_number: int) -> OrderedDict[str, Any]:
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
    ordered["val_outcome"] = doc.get("val_outcome") or {}
    return ordered


def convert_file(src: Path, dst: Path) -> int:
    rows: list[str] = []
    with src.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f, start=1):
            if not line.strip():
                continue
            doc = json.loads(line)
            ordered = _ordered_doc(doc, article_number=index)
            rows.append(json.dumps(ordered, ensure_ascii=False))

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    return len(rows)


def main() -> None:
    jobs = [
        (
            project_root / "exports" / "economics_validate_val_outcome.jsonl",
            project_root / "reports" / "data" / "model_predictions" / "subjects" / "200_econ.jsonl",
        ),
        (
            project_root / "exports" / "sociology_validate_val_outcome.jsonl",
            project_root / "reports" / "data" / "model_predictions" / "subjects" / "200_social.jsonl",
        ),
    ]

    for src, dst in jobs:
        count = convert_file(src, dst)
        print(f"{src} -> {dst} rows={count}")


if __name__ == "__main__":
    main()
