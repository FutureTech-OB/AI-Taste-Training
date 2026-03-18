"""
Article Practice Schema - 数据模式定义（BaseModel, Filter等）
"""

from .filter import ArticleFilter
from .pdf import ExtractedImage, ExtractedTable, GeneratedImage
from src.core.schema.validation import ValOutcome

__all__ = [
    "ArticleFilter",
    "ExtractedImage",
    "ExtractedTable",
    "GeneratedImage",
    "ValOutcome",
]

