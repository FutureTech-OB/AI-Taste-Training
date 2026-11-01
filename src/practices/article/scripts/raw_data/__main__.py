"""
CLI：
  - metadata.xlsx → metadata_translated.json
  - .json 数组 → .jsonl

用法:
  python -m src.practices.article.scripts.raw_data [input_path] [-o output_path]
  若 input_path 以 .json 结尾则转为 JSONL，否则按 Excel 跑 pipeline。
"""
import argparse
import logging
from pathlib import Path

from .metadata_pipeline import run_pipeline
from .json_to_jsonl import json_array_to_jsonl

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Excel → metadata_translated.json；或 JSON 数组 → JSONL"
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="assets/metadata/metadata.xlsx",
        help="输入：.xlsx 或 .json（数组）",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="输出路径（默认：Excel 同目录 metadata_translated.json；JSON 同目录同名校 .jsonl）",
    )
    args = parser.parse_args()

    path = Path(args.input_path)
    if not path.exists():
        raise SystemExit(f"文件不存在: {path}")

    if path.suffix.lower() == ".json":
        out = json_array_to_jsonl(path, args.output)
    else:
        out = run_pipeline(path, args.output)
    print(f"已生成: {out}")


if __name__ == "__main__":
    main()
