"""
MongoDB数据加载器
"""
from typing import AsyncIterator, Dict, Any, Type, Optional
from beanie import Document
from .base import BaseDataLoader
from ..schema.filter import BaseFilter, FilterField, FilterOperator


class MongoDBLoader(BaseDataLoader):
    """MongoDB流式数据加载器"""

    def __init__(self, document_class: Type[Document], id_field: str = "id"):
        """
        Args:
            document_class: Beanie Document类
            id_field: 唯一标识符字段名（如 "id", "doi", "title"）
        """
        self.document_class = document_class
        self.id_field = id_field

    async def load_stream(
        self,
        filter: BaseFilter,
        batch_size: int = 100
    ) -> AsyncIterator[Dict[str, Any]]:
        """使用MongoDB cursor流式读取"""
        query = filter.to_mongo_query()

        # 使用cursor，不调用to_list()
        cursor = self.document_class.find(query)

        async for doc in cursor:
            yield doc.model_dump()  # 逐条yield，不占用大量内存

    async def count(self, filter: BaseFilter) -> int:
        """只统计数量，不加载数据"""
        query = filter.to_mongo_query()
        return await self.document_class.find(query).count()

    async def save_item(self, item: Dict[str, Any]) -> bool:
        """
        保存或更新数据项到MongoDB
        
        Args:
            item: 数据项字典
            
        Returns:
            是否保存成功
        """
        try:
            # 获取唯一标识符
            item_id = item.get(self.id_field)
            if not item_id:
                return False
            
            # 查找现有文档
            # 注意：Beanie 的 "id" 是 "_id" 的 Python 别名，但在 raw dict 查询中不会自动转换，
            # 必须用 .get() 按 _id 查找，否则 find_one({"id": ...}) 找不到文档，
            # 导致后续 insert() 报 E11000 duplicate key error。
            if self.id_field == "id":
                existing_doc = await self.document_class.get(item_id)
            else:
                existing_doc = await self.document_class.find_one(
                    {self.id_field: item_id}
                )
            
            if existing_doc:
                # 更新现有文档
                for key, value in item.items():
                    setattr(existing_doc, key, value)
                await existing_doc.save()
            else:
                # 创建新文档
                new_doc = self.document_class(**item)
                await new_doc.insert()
            
            return True
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"保存数据项失败: {e}")
            return False

    async def find_item(self, item_id: str, id_field: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        根据ID查找数据项
        
        Args:
            item_id: 数据项唯一标识符
            id_field: ID字段名（如果为None则使用初始化时的id_field）
            
        Returns: 
        
            数据项字典，如果不存在则返回 None
        """
        field = id_field or self.id_field
        if field == "id":
            doc = await self.document_class.get(item_id)
        else:
            doc = await self.document_class.find_one({field: item_id})
        if doc:
            return doc.model_dump()
        return None
