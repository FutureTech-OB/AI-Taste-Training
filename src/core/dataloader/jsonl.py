"""
JSONL数据加载器
"""
import json
import asyncio
import aiofiles
import os
import logging
from typing import AsyncIterator, Dict, Any, Optional
from pathlib import Path
from .base import BaseDataLoader
from ..schema.filter import BaseFilter, FilterOperator

logger = logging.getLogger(__name__)


class JSONLLoader(BaseDataLoader):
    """JSONL流式数据加载器（定时自动flush）"""

    def __init__(self, file_path: str, id_field: str = "id", flush_interval: int = 10):
        """
        Args:
            file_path: JSONL文件路径
            id_field: 唯一标识符字段名（如 "id", "doi", "title"）
            flush_interval: 每保存多少个item后自动flush（默认10，设为0则禁用自动flush）
        """
        self.file_path = file_path
        self.id_field = id_field
        self.flush_interval = flush_interval
        self._cache: Dict[str, Dict[str, Any]] = {}  # 内存缓存，用于批量更新
        self._save_count = 0  # 保存计数器
        self._flush_lock = asyncio.Lock()  # 防止并发 flush 导致文件损坏

    async def load_stream(
        self,
        filter: BaseFilter,
        batch_size: int = 100
    ) -> AsyncIterator[Dict[str, Any]]:
        """逐行读取JSONL文件"""
        async with aiofiles.open(self.file_path, 'r', encoding='utf-8') as f:
            async for line in f:
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)

                    # 应用过滤器
                    if self._match_filter(data, filter):
                        yield data
                except json.JSONDecodeError:
                    continue

    async def count(self, filter: BaseFilter) -> int:
        """统计符合条件的数据数量"""
        count = 0
        async for _ in self.load_stream(filter):
            count += 1
        return count

    def _match_filter(self, data: Dict, filter: BaseFilter) -> bool:
        """检查数据是否符合过滤条件"""
        for f in filter.filters:
            value = data.get(f.name)

            if f.operator == FilterOperator.EQ and value != f.value:
                return False
            elif f.operator == FilterOperator.NE and value == f.value:
                return False
            elif f.operator == FilterOperator.IN and value not in f.value:
                return False
            elif f.operator == FilterOperator.NIN and value in f.value:
                return False
            elif f.operator == FilterOperator.GT and not (value is not None and value > f.value):
                return False
            elif f.operator == FilterOperator.GTE and not (value is not None and value >= f.value):
                return False
            elif f.operator == FilterOperator.LT and not (value is not None and value < f.value):
                return False
            elif f.operator == FilterOperator.LTE and not (value is not None and value <= f.value):
                return False
            elif f.operator == FilterOperator.CONTAINS and not (value and f.value in str(value)):
                return False

        return True

    async def save_item(self, item: Dict[str, Any]) -> bool:
        """
        保存或更新数据项到JSONL文件

        注意：JSONL文件更新需要重写整个文件，性能较低
        会根据 flush_interval 自动批量刷新缓存

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

            # 将更新项加入缓存（dict 赋值在单线程 asyncio 中是安全的）
            self._cache[item_id] = item
            self._save_count += 1

            # 达到批次阈值时自动 flush
            if self.flush_interval > 0 and self._save_count >= self.flush_interval:
                await self.flush()

            return True
        except Exception as e:
            logger.error(f"保存数据项失败: {e}")
            return False

    async def flush(self) -> bool:
        """
        将缓存中的所有更新刷新到文件（线程安全）

        使用 asyncio.Lock 保证同一时间只有一个 flush 在执行，
        避免并发读写文件导致数据损坏。

        Returns:
            是否刷新成功
        """
        async with self._flush_lock:
            if not self._cache:
                return True

            # 先快照当前缓存，然后立即清空，释放给其他协程继续写入
            pending = dict(self._cache)
            self._cache.clear()
            self._save_count = 0

            try:
                # 读取现有数据
                existing_items: Dict[str, Dict[str, Any]] = {}
                if os.path.exists(self.file_path):
                    async with aiofiles.open(self.file_path, 'r', encoding='utf-8') as f:
                        async for line in f:
                            if not line.strip():
                                continue
                            try:
                                data = json.loads(line)
                                item_id = data.get(self.id_field)
                                if item_id:
                                    existing_items[item_id] = data
                            except json.JSONDecodeError:
                                continue

                # 合并更新
                existing_items.update(pending)

                # 确保目录存在
                Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)

                # 先写到临时文件，成功后再替换，防止写入中途崩溃损坏原文件
                tmp_path = self.file_path + ".tmp"
                async with aiofiles.open(tmp_path, 'w', encoding='utf-8') as f:
                    for item_data in existing_items.values():
                        line = json.dumps(item_data, ensure_ascii=False)
                        await f.write(line + "\n")

                # 原子替换（Windows 上 os.replace 也是原子的）
                os.replace(tmp_path, self.file_path)

                logger.info(f"已 flush {len(pending)} 条到 {self.file_path}（文件共 {len(existing_items)} 条）")
                return True
            except Exception as e:
                # flush 失败：把未写入的数据放回缓存，下次重试
                self._cache.update(pending)
                logger.error(f"刷新缓存失败: {e}")
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
        
        # 先检查缓存
        if item_id in self._cache:
            return self._cache[item_id]
        
        # 从文件查找
        async with aiofiles.open(self.file_path, 'r', encoding='utf-8') as f:
            async for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if data.get(field) == item_id:
                        return data
                except json.JSONDecodeError:
                    continue
        
        return None
