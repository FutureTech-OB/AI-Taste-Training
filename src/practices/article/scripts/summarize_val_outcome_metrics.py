"""
Summarize Accuracy and Macro F1 for models stored in val_outcome JSONL files.

Prediction priority per model output:
1. prediction
2. vote_predictions / vote_counts / response_text
3. logp argmax

Example:
  python -m src.practices.article.scripts.summarize_val_outcome_metrics \
    --inputs exports/economics_validate_val_outcome.jsonl exports/sociology_validate_val_outcome.jsonl \
    --out-md reports/validate_model_metrics.md \
    --out-csv reports/validate_model_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABELS = ["exceptional", "strong", "fair", "limited"]
LABEL_SET = set(LABELS)

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def _normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text if text in LABEL_SET else None


def _prediction_from_vote_predictions(values: Any) -> str | None:
    if not isinstance(values, list):
        return None
    preds = [_normalize_label(v) for v in values]
    preds = [p for p in preds if p]
    if not preds:
        return None
    counts = Counter(preds)
    top_n = counts.most_common()
    best_count = top_n[0][1]
    tied = sorted(label for label, count in top_n if count == best_count)
    return tied[0] if tied else None


def _prediction_from_vote_counts(values: Any) -> str | None:
    if not isinstance(values, dict):
        return None
    pairs = []
    for key, value in values.items():
        label = _normalize_label(key)
        if label is None:
            continue
        try:
            count = float(value)
        except (TypeError, ValueError):
            continue
        pairs.append((label, count))
    if not pairs:
        return None
    best = max(count for _, count in pairs)
    tied = sorted(label for label, count in pairs if count == best)
    return tied[0] if tied else None


def _prediction_from_logp(values: Any) -> str | None:
    if not isinstance(values, dict):
        return None
    pairs = []
    for key, value in values.items():
        label = _normalize_label(key)
        if label is None:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        pairs.append((label, score))
    if not pairs:
        return None
    best = max(score for _, score in pairs)
    tied = sorted(label for label, score in pairs if score == best)
    return tied[0] if tied else None


def extract_prediction(model_output: dict[str, Any]) -> tuple[str | None, str | None]:
    pred = _normalize_label(model_output.get("prediction"))
    if pred:
        return pred, "prediction"

    pred = _prediction_from_vote_predictions(model_output.get("vote_predictions"))
    if pred:
        return pred, "vote_predictions"

    pred = _prediction_from_vote_counts(model_output.get("vote_counts"))
    if pred:
        return pred, "vote_counts"

    pred = _normalize_label(model_output.get("response_text"))
    if pred:
        return pred, "response_text"

    pred = _prediction_from_logp(model_output.get("logp"))
    if pred:
        return pred, "logp"

    return None, None


def compute_metrics(pairs: list[tuple[str, str]]) -> tuple[float, float]:
    if not pairs:
        return 0.0, 0.0

    total = len(pairs)
    correct = sum(1 for gt, pred in pairs if gt == pred)
    acc = 100.0 * correct / total

    f1_scores = []
    for label in LABELS:
        tp = sum(1 for gt, pred in pairs if gt == label and pred == label)
        fp = sum(1 for gt, pred in pairs if gt != label and pred == label)
        fn = sum(1 for gt, pred in pairs if gt == label and pred != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_scores.append(f1)

    macro_f1 = 100.0 * sum(f1_scores) / len(f1_scores)
    return acc, macro_f1


def summarize_file(path: Path, entry: str) -> list[dict[str, Any]]:
    model_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    model_sources: dict[str, Counter[str]] = defaultdict(Counter)
    subject_name = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            gt = _normalize_label(obj.get("rank"))
            if gt is None:
                continue
            subject_name = subject_name or obj.get("subject") or path.stem
            outputs = ((obj.get("val_outcome") or {}).get(entry) or {})
            if not isinstance(outputs, dict):
                continue
            for model_name, model_output in outputs.items():
                if not isinstance(model_output, dict):
                    continue
                pred, source = extract_prediction(model_output)
                if pred is None:
                    continue
                model_pairs[model_name].append((gt, pred))
                if source:
                    model_sources[model_name][source] += 1

    rows = []
    for model_name, pairs in model_pairs.items():
        acc, macro_f1 = compute_metrics(pairs)
        source_counts = model_sources.get(model_name, Counter())
        primary_source = source_counts.most_common(1)[0][0] if source_counts else ""
        rows.append(
            {
                "subject": subject_name or path.stem,
                "model": model_name,
                "n": len(pairs),
                "accuracy": acc,
                "macro_f1": macro_f1,
                "prediction_source": primary_source,
            }
        )

    rows.sort(key=lambda item: (-item["macro_f1"], -item["accuracy"], item["model"]))
    return rows


def write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subject", "model", "n", "accuracy", "macro_f1", "prediction_source"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "accuracy": f"{row['accuracy']:.2f}",
                    "macro_f1": f"{row['macro_f1']:.2f}",
                }
            )


def write_md(grouped_rows: dict[str, list[dict[str, Any]]], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Validate Model Results",
        "",
    ]

    for subject, rows in grouped_rows.items():
        lines.extend(
            [
                f"## {subject}",
                "",
                "| Model | N | Accuracy | Macro F1 | Source |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['model']} | {row['n']} | {row['accuracy']:.2f}% | {row['macro_f1']:.2f}% | {row['prediction_source']} |"
            )
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Accuracy and Macro F1 from val_outcome JSONL files"
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSONL files")
    parser.add_argument("--entry", default="rq_with_context", help="val_outcome entry key")
    parser.add_argument("--out-md", default=None, help="Optional markdown report path")
    parser.add_argument("--out-csv", default=None, help="Optional csv report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(item) for item in args.inputs]

    all_rows: list[dict[str, Any]] = []
    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    for path in input_paths:
        rows = summarize_file(path, args.entry)
        if not rows:
            continue
        subject = rows[0]["subject"]
        grouped_rows[subject] = rows
        all_rows.extend(rows)

    for subject, rows in grouped_rows.items():
        print(f"[{subject}]")
        for row in rows:
            print(
                f"{row['model']}\tN={row['n']}\tAcc={row['accuracy']:.2f}%\tMacroF1={row['macro_f1']:.2f}%\tSource={row['prediction_source']}"
            )
        print()

    if args.out_csv:
        write_csv(all_rows, Path(args.out_csv))
        print(f"out_csv={args.out_csv}")
    if args.out_md:
        write_md(grouped_rows, Path(args.out_md))
        print(f"out_md={args.out_md}")


if __name__ == "__main__":
    main()
