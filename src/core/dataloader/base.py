"""
抽象数据加载器基类
"""
from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Any, Optional
from ..schema.filter import BaseFilter


class BaseDataLoader(ABC):
    """抽象数据加载器 - 支持流式处理和保存"""

    @abstractmethod
    async def load_stream(
        self,
        filter: BaseFilter,
        batch_size: int = 100
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        流式加载数据（异步生成器）

        Args:
            filter: 过滤条件
            batch_size: 批处理大小（用于优化，不影响输出）

        Yields:
            单条数据记录
        """
        pass

    @abstractmethod
    async def count(self, filter: BaseFilter) -> int:
        """
        获取数据总数（不加载数据）

        Args:
            filter: 过滤条件

        Returns:
            符合条件的数据总数
        """
        pass

    @abstractmethod
    async def save_item(self, item: Dict[str, Any]) -> bool:
        """
        保存或更新数据项

        Args:
            item: 数据项字典（必须包含唯一标识符）

        Returns:
            是否保存成功
        """
        pass

    async def find_item(self, item_id: str, id_field: str = "id") -> Optional[Dict[str, Any]]:
        """
        根据ID查找数据项（可选实现）

        Args:
            item_id: 数据项唯一标识符
            id_field: ID字段名（如 "id", "doi", "title"）

        Returns:
            数据项字典，如果不存在则返回 None
        """
        # 默认实现：通过 filter 查找
        from ..schema.filter import FilterField, FilterOperator
        filter = BaseFilter()
        filter.add_filter(id_field, FilterOperator.EQ, item_id)
        
        async for item in self.load_stream(filter):
            return item
        return None

    async def load_batch_stream(
        self,
        filter: BaseFilter,
        batch_size: int = 100
    ) -> AsyncIterator[list]:
        """
        流式加载数据（批量模式）

        Args:
            filter: 过滤条件
            batch_size: 每批数量

        Yields:
            批量数据列表
        """
        batch = []
        async for item in self.load_stream(filter, batch_size):
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:  # 最后一批
            yield batch
