"""
Article数据加载器
"""
from typing import AsyncIterator, Dict, Any
from src.core.dataloader import MongoDBLoader
from .models import Article
from .schema import ArticleFilter


class ArticleLoader(MongoDBLoader):
    """Article数据加载器 - 直接继承MongoDBLoader"""

    def __init__(self):
        """初始化Article加载器，绑定Article模型"""
        super().__init__(Article)
