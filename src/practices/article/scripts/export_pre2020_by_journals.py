"""
Export pre-2020 Article docs for selected journals into JSONL for SFT (data_source=jsonl).

Filters:
- published_year <= year_max (default 2019)
- journal in journals_file (one name per line)
- entries.<entry> has non-empty text (regex \S)
- optional subject / type

Example:
  python -m src.practices.article.scripts.export_pre2020_by_journals \
    --connection_string "mongodb://admin:admin123@localhost:27019/Article?authSource=admin" \
    --db_name Article \
    --journals_file timeline/journals_full.txt \
    --entry rq_with_context \
    --subject ob \
    --types study \
    --out_jsonl timeline/pre2020_train.jsonl
"""
import argparse
from pathlib import Path
from typing import List, Optional
import asyncio
import json

from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

from src.practices.article.models import Article
from src.practices.article.utils_rank import normalize_rank
from src.practices.article.scripts.fill_rank_by_journal import (
    _normalize_journal_name as _norm_journal,
    _build_rank_mapping as _build_rank_map,
)


def load_journals(path: Path) -> List[str]:
    items: List[str] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s:
                items.append(s)
    return items


async def export(
    connection_string: str,
    db_name: str,
    journals_file: Path,
    out_jsonl: Path,
    entry: str,
    subject: Optional[str] = None,
    types: Optional[str] = None,
    year_max: int = 2019,
    assign_rank_by_journal: bool = False,
    overwrite_rank: bool = False,
):
    journals = load_journals(journals_file)
    if not journals:
        raise ValueError(f"Empty journals file: {journals_file}")

    client = AsyncIOMotorClient(connection_string)
    db = client[db_name]
    await init_beanie(database=db, document_models=[Article])

    q = {
        "published_year": {"$lte": int(year_max)},
        "journal": {"$in": journals},
        f"entries.{entry}": {"$regex": "\\S"},
    }
    if subject:
        q["subject"] = subject
    if types:
        q["type"] = types

    coll = Article.get_motor_collection()
    cursor = coll.find(q)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    rank_map = _build_rank_map() if assign_rank_by_journal else None
    
    def _sanitize(x):
        import datetime
        try:
            from bson import ObjectId  # type: ignore
        except Exception:
            ObjectId = None  # type: ignore
        try:
            from bson import DBRef  # type: ignore
        except Exception:
            DBRef = None  # type: ignore
        if isinstance(x, dict):
            return {k: _sanitize(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_sanitize(i) for i in x]
        if ObjectId is not None and isinstance(x, ObjectId):
            return str(x)
        if DBRef is not None and isinstance(x, DBRef):
            # drop or stringify DBRef; keep None to avoid leaking internal refs
            return None
        if isinstance(x, (datetime.datetime, datetime.date)):
            try:
                return x.isoformat()
            except Exception:
                return str(x)
        return x
    with out_jsonl.open('w', encoding='utf-8') as f:
        async for doc in cursor:
            doc.pop('_id', None)
            if not doc.get('split'):
                doc['split'] = 'train'
            # normalize rank aliases to canonical strings if possible
            rn = normalize_rank(doc.get('rank'))
            if rn:
                doc['rank'] = rn
            # optionally assign rank by journal mapping (alias-aware)
            if assign_rank_by_journal and rank_map is not None:
                current_rank = doc.get('rank')
                if overwrite_rank or not current_rank:
                    jn = _norm_journal(doc.get('journal'))
                    mapped = rank_map.get(jn)
                    if mapped:
                        doc['rank'] = mapped
            f.write(json.dumps(_sanitize(doc), ensure_ascii=False) + '\n')
            count += 1
    print(f"Exported {count} docs to {out_jsonl}")


def main():
    ap = argparse.ArgumentParser(description='Export pre-2020 journals to JSONL')
    ap.add_argument('--connection_string', required=True)
    ap.add_argument('--db_name', required=True)
    ap.add_argument('--journals_file', default='timeline/journals_full.txt')
    ap.add_argument('--out_jsonl', default='timeline/pre2020_train.jsonl')
    ap.add_argument('--entry', default='rq_with_context')
    ap.add_argument('--subject', default=None)
    ap.add_argument('--types', default=None)
    ap.add_argument('--year_max', type=int, default=2019)
    ap.add_argument('--assign_rank_by_journal', action='store_true', help='根据期刊名映射设置 rank（处理别名）')
    ap.add_argument('--overwrite_rank', action='store_true', help='覆盖已有 rank')
    args = ap.parse_args()

    asyncio.run(export(
        connection_string=args.connection_string,
        db_name=args.db_name,
        journals_file=Path(args.journals_file),
        out_jsonl=Path(args.out_jsonl),
        entry=args.entry,
        subject=args.subject,
        types=args.types,
        year_max=args.year_max,
        assign_rank_by_journal=args.assign_rank_by_journal,
        overwrite_rank=args.overwrite_rank,
    ))


if __name__ == '__main__':
    main()
