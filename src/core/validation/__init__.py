"""
Core验证模块 - 基于Message的通用验证功能
"""
from .validator import BaseValidator
from .metrics import calculate_metrics, save_results

__all__ = [
    "BaseValidator",
    "calculate_metrics",
    "save_results",
]
