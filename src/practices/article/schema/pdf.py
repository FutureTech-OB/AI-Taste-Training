"""
PDF数据模式定义
"""
import datetime
from pydantic import BaseModel, Field
from typing import Optional, Tuple


class ExtractedImage(BaseModel):
    """提取的图片"""
    size: Tuple[int, int]
    title: Optional[str] = None
    url: Optional[str] = None
    description: Optional[str] = None
    context: Optional[str] = None
    page: Optional[int] = None
    data: Optional[bytes] = None

    def format_metadata(self):
        metadata = ""
        for key, value in self.model_dump().items():
            if value:
                metadata += f"{key}: {value}\n"
        return metadata


class ExtractedTable(BaseModel):
    """提取的表格"""
    table: str
    page: int
    title: Optional[str] = None

    def format_metadata(self):
        return f"Table content: {self.table}\n"


class GeneratedImage(BaseModel):
    """生成的图片"""
    title: str
    url: str
    size: Tuple[int, int]
    metadata: Optional[dict] = None
    generated_from: Optional[str] = None
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)

    class Settings:
        indexes = [
            "session_id",
        ]
    
    def format_metadata(self):
        metadata = ""
        metadata += f"Title: {self.title}\n"
        metadata += f"URL: {self.url}\n"
        metadata += f"Size: {self.size}\n"
        metadata += f"Priority: {self.metadata.get('priority', 'low')}\n"
        metadata += f"Paper_section: {self.metadata.get('paper_section', '')}\n"
        metadata += f"Description: {self.metadata.get('detailed_description', self.metadata.get('description', ''))}\n"

        return metadata

