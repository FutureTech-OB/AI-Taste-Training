"""
Article Practice Models - 数据库模型（Document）
"""
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
from .article import Article
from .pdf import PDFData


async def init_database(
    connection_string: str,
    db_name: Optional[str] = None
):
    """
    初始化数据库连接
    
    Args:
        connection_string: MongoDB 连接字符串
        db_name: 数据库名称（可选，如果连接字符串中已包含则不需要）
    
    Example:
        >>> await init_database("mongodb://localhost:27017/RQ")
        >>> await init_database("mongodb://localhost:27017", db_name="RQ")
    """
    client = AsyncIOMotorClient(connection_string)
    
    # 从连接字符串中提取数据库名（如果未提供 db_name）
    if db_name is None:
        if '/' in connection_string:
            # 提取连接字符串中的数据库名
            db_part = connection_string.split('/')[-1].split('?')[0]
            db_name = db_part if db_part else 'RQ'
        else:
            db_name = 'RQ'
    
    database = client[db_name]
    await init_beanie(
        database=database,
        document_models=[Article, PDFData]
    )


__all__ = [
    "Article",
    "PDFData",
    "init_database",
]

