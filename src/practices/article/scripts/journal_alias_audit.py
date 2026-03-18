"""
审计 journals_full.txt 中的期刊是否能被期刊别名映射命中，并给出 rank。

输出：
- timeline/journal_rank_mapping.csv: journal,normalized,rank,status
- timeline/journals_unmatched.txt: 未命中的期刊列表
"""
import argparse
from pathlib import Path
import csv

from src.practices.article.scripts.fill_rank_by_journal import (
    _normalize_journal_name as _norm,
    _build_rank_mapping as _build_map,
)


def load_journals(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s:
                yield s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--journals_file', default='timeline/journals_full.txt')
    ap.add_argument('--out_csv', default='timeline/journal_rank_mapping.csv')
    ap.add_argument('--out_unmatched', default='timeline/journals_unmatched.txt')
    args = ap.parse_args()

    journals_path = Path(args.journals_file)
    out_csv = Path(args.out_csv)
    out_unmatched = Path(args.out_unmatched)

    mapping = _build_map()
    rows = []
    unmatched = []
    for j in load_journals(journals_path):
        key = _norm(j)
        rank = mapping.get(key)
        if rank:
            rows.append((j, key, rank, 'matched'))
        else:
            rows.append((j, key, '', 'unmatched'))
            unmatched.append(j)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['journal', 'normalized', 'rank', 'status'])
        w.writerows(rows)

    if unmatched:
        with out_unmatched.open('w', encoding='utf-8') as f:
            for j in unmatched:
                f.write(j + '\n')

    print(f"Wrote CSV: {out_csv}  (rows={len(rows)})")
    print(f"Unmatched: {len(unmatched)} → {out_unmatched}")


if __name__ == '__main__':
    main()

