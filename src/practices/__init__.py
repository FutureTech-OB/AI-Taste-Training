"""
Practices - 具体业务实现层
"""
from .registry import PracticeRegistry

# 可选导入验证模块
try:
    from .validation import validate, article_to_messages, process_article_messages, calculate_metrics, save_results
    __all__ = [
        "PracticeRegistry",
        "validate",
        "article_to_messages",
        "process_article_messages",
        "calculate_metrics",
        "save_results",
    ]
except ImportError:
    __all__ = [
        "PracticeRegistry",
    ]
