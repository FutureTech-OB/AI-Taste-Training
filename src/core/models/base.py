"""
数据库基础模型
"""
from beanie import Document
from datetime import datetime
from pydantic import Field


class BaseDocument(Document):
    """基础文档模型"""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Settings:
        # 子类需要覆盖
        name = "base_documents"
        is_root = True

    def update_timestamp(self):
        """更新时间戳"""
        self.updated_at = datetime.now()
