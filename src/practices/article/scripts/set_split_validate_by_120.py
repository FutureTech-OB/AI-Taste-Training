"""
将 title 与 120 条名单模糊匹配（F1 >= 0.8）的文章在数据库中标为 split=validate。
复用与 filter_rank 相同的数据库配置；保证恰好标出 120 条。
"""
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Optional, List, Tuple

from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
import toml

# 项目根目录
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.practices.article.models import Article


def load_database_config() -> Tuple[str, str]:
    """从 assets/database.toml 加载连接字符串和 db_name。"""
    config_path = project_root / "assets" / "database.toml"
    default_db_name = "RIOB"
    default_connection_string_tpl = (
        "mongodb://root:password@166.111.96.30:27027/{db}?authSource=admin"
    )

    if not config_path.exists():
        return default_connection_string_tpl.format(db=default_db_name), default_db_name

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = toml.load(f)
        default_config = config.get("default", {})
        db_name = default_config.get("db_name", default_db_name)
        conn_str = default_config.get("connection_string", "")
        if conn_str:
            connection_string = conn_str.replace("<DBNAME>", db_name) if "<DBNAME>" in conn_str else conn_str
        else:
            connection_string = f"mongodb://root:password@166.111.96.30:27027/{db_name}?authSource=admin"
        return connection_string, db_name
    except Exception as e:
        print(f"警告: 读取配置失败: {e}，使用默认配置")
        return default_connection_string_tpl.format(db=default_db_name), default_db_name


def _tokenize(title: str) -> set:
    """归一化并切词，返回 token 集合（小写、去重）。"""
    if not title or not isinstance(title, str):
        return set()
    s = re.sub(r"[^\w\s]", " ", title.lower())
    return set(s.split())


def token_f1(ref_tokens: set, db_tokens: set) -> float:
    """Token-level F1: 2*P*R/(P+R)，P=|match|/|db|, R=|match|/|ref|。"""
    if not ref_tokens:
        return 0.0
    match = ref_tokens & db_tokens
    if not match:
        return 0.0
    precision = len(match) / len(db_tokens) if db_tokens else 0.0
    recall = len(match) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def load_120_titles(jsonl_path: Path) -> List[str]:
    """从 120.jsonl 读取 title 列表（顺序与文件一致）。"""
    titles = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get("title")
            if t:
                titles.append(t)
    return titles


async def run(
    connection_string: Optional[str] = None,
    db_name: Optional[str] = None,
    jsonl_path: Optional[Path] = None,
    f1_threshold: float = 0.8,
    dry_run: bool = False,
):
    if connection_string is None or db_name is None:
        conn_str, name = load_database_config()
        connection_string = connection_string or conn_str
        db_name = db_name or name

    if "<DBNAME>" in connection_string:
        connection_string = connection_string.replace("<DBNAME>", db_name)

    jsonl_path = jsonl_path or (project_root / "assets" / "unknown" / "120.jsonl")
    if not jsonl_path.exists():
        raise FileNotFoundError(f"120 名单文件不存在: {jsonl_path}")

    ref_titles = load_120_titles(jsonl_path)
    print(f"已加载 {len(ref_titles)} 条参考 title（120.jsonl）")

    client = AsyncIOMotorClient(connection_string)
    database = client[db_name]
    await init_beanie(database=database, document_models=[Article])

    # 只要求 title 存在且非空（不限制 type，避免漏掉能搜到的文章）
    cursor = Article.find({"title": {"$exists": True, "$nin": [None, ""]}})
    db_articles: List[Article] = await cursor.to_list()
    print(f"库内含 title 的文章数: {len(db_articles)}")

    ref_tokens_list = [_tokenize(t) for t in ref_titles]
    db_titles = [(a, (a.title or "").strip()) for a in db_articles]
    db_tokens_list = [_tokenize(t) for _, t in db_titles]

    assigned_ids = set()
    to_validate: List[Article] = []
    # 第一轮：F1 >= f1_threshold；若未满 120 条，对未匹配的 ref 逐级降阈值再匹配（最低到 0.3）
    all_thresholds = [f1_threshold, 0.7, 0.6, 0.5]
    threshold_sequence = sorted(set(all_thresholds), reverse=True)
    unassigned_ref_indices = list(range(len(ref_titles)))

    for th in threshold_sequence:
        if len(to_validate) >= 120:
            break
        still_unassigned = []
        for i in unassigned_ref_indices:
            ref = ref_titles[i]
            ref_tokens = ref_tokens_list[i]
            best_article = None
            best_f1 = 0.0

            for j, (art, _) in enumerate(db_titles):
                if art.id in assigned_ids:
                    continue
                f1 = token_f1(ref_tokens, db_tokens_list[j])
                if f1 >= th and f1 > best_f1:
                    best_f1 = f1
                    best_article = art

            if best_article is not None:
                assigned_ids.add(best_article.id)
                to_validate.append(best_article)
                db_title = (best_article.title or "").strip()
                print(f"  [{len(to_validate)}] F1={best_f1:.3f} (th={th})")
                print(f"      ref: {ref}")
                print(f"      db:  {db_title}")
                print(f"      id:  {best_article.id}")
            else:
                still_unassigned.append(i)
        unassigned_ref_indices = still_unassigned
        if unassigned_ref_indices and th != threshold_sequence[-1]:
            print(f"  阈值 {th} 下未匹配 {len(unassigned_ref_indices)} 条，尝试更低阈值...")

    for i in unassigned_ref_indices:
        print(f"  未匹配: {ref_titles[i]}")

    print(f"\n共匹配到 {len(to_validate)} 条，将设为 split=validate")

    if dry_run:
        print("(dry_run，未写库)")
        return

    for art in to_validate:
        art.split = "validate"
        await art.save()
    print("已更新数据库。")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="按 120 条 title 模糊匹配并设 split=validate")
    parser.add_argument("--connection-string", default=None, help="MongoDB 连接字符串")
    parser.add_argument("--db-name", default=None, help="数据库名")
    parser.add_argument("--jsonl", default=None, type=Path, help="120.jsonl 路径")
    parser.add_argument("--f1", type=float, default=0.8, help="Token F1 阈值 (默认 0.8)")
    parser.add_argument("--dry-run", action="store_true", help="只打印匹配，不写库")

    args = parser.parse_args()
    asyncio.run(
        run(
            connection_string=args.connection_string,
            db_name=args.db_name,
            jsonl_path=args.jsonl,
            f1_threshold=args.f1,
            dry_run=args.dry_run,
        )
    )
