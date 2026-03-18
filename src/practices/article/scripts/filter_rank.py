"""
统计每个学科每个rank的study类型文章总数量
按学科排序显示
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


async def filter_rank(
    connection_string: Optional[str] = None,
    db_name: Optional[str] = None
):
    """
    统计每个学科每个rank的study类型文章总数量
    
    Args:
        connection_string: MongoDB连接字符串
        db_name: 数据库名称
    """
    # 加载配置
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
    print(f"数据库名称: {db_name}\n")
    
    # 初始化数据库连接
    client = AsyncIOMotorClient(connection_string)
    database = client[db_name]
    
    await init_beanie(
        database=database,
        document_models=[Article]
    )
    
    # 使用MongoDB聚合查询统计
    # 按学科和rank分组统计study类型文章数量
    pipeline = [
        {"$match": {"type": "study"}},
        {
            "$group": {
                "_id": {
                    "subject": "$subject",
                    "rank": "$rank"
                },
                "article_count": {"$sum": 1}
            }
        },
        {
            "$project": {
                "subject": "$_id.subject",
                "rank": "$_id.rank",
                "article_count": 1,
                "_id": 0
            }
        },
        {"$sort": {"subject": 1, "rank": 1}}
    ]
    
    print("正在统计各学科各rank的study类型文章数量...\n")
    results = await Article.aggregate(pipeline).to_list()
    
    # 按学科分组
    by_subject = {}
    for r in results:
        subject = r.get('subject') or 'None'
        rank = r.get('rank') or 'None'
        count = r.get('article_count', 0)
        
        if subject not in by_subject:
            by_subject[subject] = {}
        by_subject[subject][rank] = count
    
    # 定义rank顺序
    rank_order = ['exceptional', 'strong', 'fair', 'limited', 'None']
    
    # 按学科排序显示
    print("=" * 100)
    print(f"{'学科':<30} {'exceptional':<15} {'strong':<15} {'fair':<15} {'limited':<15} {'总计':<15}")
    print("=" * 100)
    
    total_by_rank = {'exceptional': 0, 'strong': 0, 'fair': 0, 'limited': 0, 'None': 0}
    grand_total = 0
    
    for subject in sorted(by_subject.keys()):
        subject_data = by_subject[subject]
        subject_total = sum(subject_data.values())
        grand_total += subject_total
        
        # 按rank顺序显示
        row = [subject[:28]]
        for rank in rank_order:
            count = subject_data.get(rank, 0)
            row.append(f"{count:<15}")
            total_by_rank[rank] += count
        row.append(f"{subject_total:<15}")
        
        print("  ".join(row))
    
    # 显示总计
    print("-" * 100)
    total_row = ["总计"]
    for rank in rank_order:
        total_row.append(f"{total_by_rank[rank]:<15}")
    total_row.append(f"{grand_total:<15}")
    print("  ".join(total_row))
    print("=" * 100)
    
    # 显示每个学科的详细信息
    print("\n详细信息（按学科排序）:")
    print("=" * 100)
    
    for subject in sorted(by_subject.keys()):
        subject_data = by_subject[subject]
        subject_total = sum(subject_data.values())
        
        print(f"\n学科: {subject}")
        print(f"  总计: {subject_total} 篇文章")
        print(f"  按Rank分布:")
        
        for rank in rank_order:
            if rank in subject_data:
                count = subject_data[rank]
                percentage = (count / subject_total * 100) if subject_total > 0 else 0
                print(f"    {rank:<15}: {count:<8} 篇 ({percentage:>6.2f}%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="统计每个学科每个rank的study类型文章总数量")
    parser.add_argument("--connection-string", help="MongoDB连接字符串（可选）")
    parser.add_argument("--db-name", help="数据库名称（可选）")
    
    args = parser.parse_args()
    
    asyncio.run(filter_rank(
        args.connection_string,
        args.db_name
    ))

