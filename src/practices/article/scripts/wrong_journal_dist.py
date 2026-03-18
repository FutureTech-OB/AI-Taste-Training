"""
从验证结果中找出错题（预测 != 标注），并基于文章 JSONL 提取其期刊分布。

用法：
  python -m src.practices.article.scripts.wrong_journal_dist \
    --validation outcome/ob_rqcontext_ob/ft_gpt-4.1-2025-04-14_personal_ob-ob-rqcontext_DHT2FqL8/validation_results.jsonl \
    --articles assets/ob/120qwen.jsonl

输出：按错误条目数对期刊排序并打印；若有未匹配上的标题，也会列出数量与样例。
"""
import argparse
import json
from pathlib import Path
from typing import Dict
from collections import Counter


def load_title_to_journal(path: Path) -> Dict[str, str]:
    m: Dict[str, str] = {}
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            title = (obj.get('title') or '').strip()
            journal = (obj.get('journal') or '').strip()
            if title:
                m[title] = journal
    return m


def load_journal_totals(path: Path) -> Counter:
    totals = Counter()
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            journal = (obj.get('journal') or '').strip()
            if journal:
                totals[journal] += 1
    return totals


def main():
    parser = argparse.ArgumentParser(description="统计错题期刊分布")
    parser.add_argument('--validation', required=True, help='validation_results.jsonl 路径')
    parser.add_argument('--articles', required=True, help='包含 title/journal 的 JSONL 路径')
    parser.add_argument('--out_csv', default=None, help='将结果导出为 CSV：journal,count,percent')
    args = parser.parse_args()

    val_path = Path(args.validation)
    art_path = Path(args.articles)
    title2journal = load_title_to_journal(art_path)
    totals_by_journal = load_journal_totals(art_path)

    wrong = Counter()
    missing = []
    total_wrong = 0

    with val_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            gt = obj.get('ground_truth')
            pred = obj.get('top1_prediction')
            title = (obj.get('item_id') or '').strip()
            if gt is None or pred is None or not title:
                continue
            if str(gt).lower() != str(pred).lower():
                total_wrong += 1
                j = title2journal.get(title)
                if j:
                    wrong[j] += 1
                else:
                    missing.append(title)

    print(f"Wrong items: {total_wrong}")
    print("期刊分布（错题数 + 期刊内占比，降序按错题数）：")
    rows = []
    for j, n in wrong.most_common():
        total_in_journal = totals_by_journal.get(j, 0)
        pct_in_journal = (n / total_in_journal * 100.0) if total_in_journal else 0.0
        rows.append((j, n, total_in_journal, pct_in_journal))
        print(f"{j}\t{n}\t(期刊内占比 {pct_in_journal:.2f}% of {total_in_journal})")

    # 导出 CSV（可选）
    if args.out_csv:
        import csv
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["journal", "wrong_count", "total_in_journal", "wrong_percent_in_journal"])
            for j, n, total_in_journal, pct_in_journal in rows:
                writer.writerow([j, n, total_in_journal, f"{pct_in_journal:.4f}"])
        print(f"\n已导出 CSV: {out_path}")
    if missing:
        print(f"\n未匹配到期刊的标题数：{len(missing)}（示例）")
        for t in missing[:10]:
            print(f"- {t}")


if __name__ == '__main__':
    main()
