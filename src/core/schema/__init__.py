"""
Core Schema - 通用数据模式定义
"""
from .message import Message, MessageRole
from .training import TrainingConfig, JobStatus, TrainingProgress
from .validation import ValidationConfig, ValidationStatus, ValidationMetrics, ValOutcome, QA
from .filter import FilterOperator, FilterField, BaseFilter
from .common import PaginationParams, PaginationResponse

__all__ = [
    "Message",
    "MessageRole",
    "TrainingConfig",
    "JobStatus",
    "TrainingProgress",
    "ValidationConfig",
    "ValidationStatus",
    "ValidationMetrics",
    "ValOutcome",
    "QA",
    "FilterOperator",
    "FilterField",
    "BaseFilter",
    "PaginationParams",
    "PaginationResponse",
]
