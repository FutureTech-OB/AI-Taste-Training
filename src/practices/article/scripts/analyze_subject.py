"""
分析指定学科的统计信息
按rank和type分组，统计journal数量和article数量
"""
import asyncio
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
import toml

# 添加项目根目录到路径（确保可以导入 src.core 和 src.practices）
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent  # 项目根目录
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.practices.article.models import Article


def load_database_config() -> Tuple[str, str]:
    """
    从配置文件加载数据库配置
    
    Returns:
        (connection_string, db_name) 元组
    """
    config_path = project_root / 'assets' / 'database.toml'
    
    # 默认值
    default_db_name = 'RItest'
    default_connection_string = f"mongodb://root:password@166.111.96.30:27027/{default_db_name}?authSource=admin"
    
    # 读取配置文件
    if not config_path.exists():
        return default_connection_string, default_db_name
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = toml.load(f)
        
        default_config = config.get('default', {})
        
        # 读取 db_name
        db_name = default_config.get('db_name', default_db_name)
        
        # 读取连接字符串
        conn_str = default_config.get('connection_string', '')
        
        # 处理连接字符串
        if conn_str:
            if '<DBNAME>' in conn_str:
                # 替换 <DBNAME> 占位符
                connection_string = conn_str.replace('<DBNAME>', db_name)
            else:
                # 如果连接字符串中没有占位符，直接使用
                connection_string = conn_str
        else:
            # 如果没有配置连接字符串，使用默认格式
            connection_string = f"mongodb://root:password@166.111.96.30:27027/{db_name}?authSource=admin"
        
        return connection_string, db_name
    
    except Exception as e:
        print(f"警告: 读取配置文件失败: {e}，使用默认配置")
        return default_connection_string, default_db_name


async def analyze_subject(
    subject: str,
    connection_string: Optional[str] = None,
    db_name: Optional[str] = None
):
    """
    分析指定学科的统计信息
    
    Args:
        subject: 学科名称（如"ECONOMICS"）
        connection_string: MongoDB连接字符串（可选，覆盖配置文件）
        db_name: 数据库名称（可选，覆盖配置文件）
    """
    # 加载配置（如果未提供参数）
    if connection_string is None or db_name is None:
        config_conn_str, config_db_name = load_database_config()
        
        if connection_string is None:
            connection_string = config_conn_str
        if db_name is None:
            db_name = config_db_name
    
    # 如果连接字符串中还有 <DBNAME>，替换它
    if '<DBNAME>' in connection_string:
        connection_string = connection_string.replace('<DBNAME>', db_name)
    
    print(f"连接数据库: {connection_string}")
    print(f"数据库名称: {db_name}")
    print(f"分析学科: {subject}\n")
    
    # 初始化数据库连接
    # 从连接字符串中提取数据库名称（如果连接字符串包含数据库名）
    # 但优先使用 db_name 参数
    client = AsyncIOMotorClient(connection_string)
    database = client[db_name]
    
    await init_beanie(
        database=database,
        document_models=[Article]
    )
    
    # 使用MongoDB聚合查询统计 - 只统计 type='study' 的文章
    # 先统计所有文章（包括journal为None的）
    pipeline_all = [
        {"$match": {"subject": subject, "type": "study"}},
        {
            "$group": {
                "_id": {
                    "type": "$type",
                    "rank": "$rank"
                },
                "article_count": {"$sum": 1}
            }
        }
    ]
    
    # 统计有journal的文章和期刊数量
    pipeline_journals = [
        {"$match": {"subject": subject, "type": "study", "journal": {"$ne": None}}},
        {
            "$group": {
                "_id": {
                    "type": "$type",
                    "rank": "$rank"
                },
                "journals": {"$addToSet": "$journal"}
            }
        },
        {
            "$project": {
                "type": "$_id.type",
                "rank": "$_id.rank",
                "journal_count": {"$size": "$journals"}
            }
        }
    ]
    
    # 按期刊统计文章数（精确到每个期刊）
    pipeline_journal_details = [
        {"$match": {"subject": subject, "type": "study", "journal": {"$ne": None}}},
        {
            "$group": {
                "_id": {
                    "journal": "$journal",
                    "rank": "$rank"
                },
                "article_count": {"$sum": 1}
            }
        },
        {
            "$project": {
                "journal": "$_id.journal",
                "rank": "$_id.rank",
                "article_count": 1
            }
        },
        {"$sort": {"rank": 1, "journal": 1}}
    ]
    
    results_all = await Article.aggregate(pipeline_all).to_list()
    results_journals = await Article.aggregate(pipeline_journals).to_list()
    journal_details = await Article.aggregate(pipeline_journal_details).to_list()
    
    # 合并结果
    results_dict = {}
    for r in results_all:
        key = (r['_id'].get('type'), r['_id'].get('rank'))
        results_dict[key] = {
            'type': r['_id'].get('type'),
            'rank': r['_id'].get('rank'),
            'article_count': r['article_count'],
            'journal_count': 0,
            'journal_details': []  # 存储每个期刊的详细信息
        }
    
    for r in results_journals:
        key = (r.get('type'), r.get('rank'))
        if key in results_dict:
            results_dict[key]['journal_count'] = r['journal_count']
    
    # 按 rank 分组期刊详情
    journal_details_by_rank = {}
    for detail in journal_details:
        rank = detail.get('rank') or 'None'
        if rank not in journal_details_by_rank:
            journal_details_by_rank[rank] = []
        journal_details_by_rank[rank].append({
            'journal': detail.get('journal'),
            'article_count': detail['article_count']
        })
    
    # 将期刊详情添加到结果中
    for key, result in results_dict.items():
        rank = result.get('rank') or 'None'
        if rank in journal_details_by_rank:
            result['journal_details'] = journal_details_by_rank[rank]
    
    results = list(results_dict.values())
    
    # 按type和rank排序
    results.sort(key=lambda x: (x.get('type') or '', x.get('rank') or ''))
    
    # 统计总数 - 只统计 type='study' 的文章
    total_articles = await Article.find({"subject": subject, "type": "study"}).count()
    # 只统计有journal的记录 - 使用get_motor_collection()获取底层collection
    collection = Article.get_motor_collection()
    journals_list = await collection.distinct("journal", {"subject": subject, "type": "study", "journal": {"$ne": None}})
    total_journals = len(journals_list)
    
    # 按type分组显示
    print("=" * 80)
    print(f"学科: {subject}")
    print("=" * 80)
    print(f"\n总计:")
    print(f"  文章总数: {total_articles}")
    print(f"  期刊总数: {total_journals}")
    if total_journals > 0:
        avg_articles_per_journal = total_articles / total_journals
        print(f"  平均每期刊文章数: {avg_articles_per_journal:.2f}")
    print()
    
    # 只显示 study 类型
    print(f"\n类型: study")
    print("-" * 80)
    print(f"  文章总数: {total_articles}")
    print(f"  期刊总数: {total_journals}")
    if total_journals > 0:
        avg_articles_per_journal = total_articles / total_journals
        print(f"  平均每期刊文章数: {avg_articles_per_journal:.2f}")
    print()
    print(f"  按Rank分布:")
    print(f"    {'Rank':<15} {'期刊数':<10} {'文章数':<10} {'每期刊文章数':<15}")
    print(f"    {'-'*15} {'-'*10} {'-'*10} {'-'*15}")
    
    for result in results:
        rank = result.get('rank') or 'None'
        journal_count = result['journal_count']
        article_count = result['article_count']
        avg_per_journal = article_count / journal_count if journal_count > 0 else 0
        avg_str = f"{avg_per_journal:.2f}" if journal_count > 0 else "N/A"
        print(f"    {rank:<15} {journal_count:<10} {article_count:<10} {avg_str:<15}")
    
    # 显示每个期刊的详细信息
    print(f"\n  按期刊详细统计:")
    print(f"    {'Rank':<15} {'期刊名称':<50} {'文章数':<10}")
    print(f"    {'-'*15} {'-'*50} {'-'*10}")
    
    for result in results:
        rank = result.get('rank') or 'None'
        journal_details = result.get('journal_details', [])
        # 按文章数降序排序
        journal_details_sorted = sorted(journal_details, key=lambda x: x['article_count'], reverse=True)
        for detail in journal_details_sorted:
            journal_name = detail.get('journal') or 'Unknown'
            article_count = detail['article_count']
            print(f"    {rank:<15} {journal_name[:48]:<50} {article_count:<10}")
    
    # 如果没有结果
    if not results:
        print(f"\n未找到学科 '{subject}' 的数据")
        # 显示可用的学科
        print("\n可用的学科列表:")
        collection = Article.get_motor_collection()
        subjects = await collection.distinct("subject")
        for subj in sorted(subjects):
            if subj:
                count = await Article.find({"subject": subj}).count()
                print(f"  {subj}: {count} 篇文章")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="分析指定学科的统计信息")
    parser.add_argument("subject", help="学科名称（如ECONOMICS）")
    parser.add_argument("--connection-string", help="MongoDB连接字符串（可选）")
    parser.add_argument("--db-name", help="数据库名称（可选）")
    
    args = parser.parse_args()
    
    asyncio.run(analyze_subject(
        args.subject,
        args.connection_string,
        args.db_name
    ))

