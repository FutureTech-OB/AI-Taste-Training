"""
Core Models - 通用数据库模型
"""
from .base import BaseDocument
from .finetune import FineTuneJob

__all__ = [
    "BaseDocument",
    "FineTuneJob",
]
