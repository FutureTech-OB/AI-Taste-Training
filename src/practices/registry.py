"""
Practice注册中心 - 管理不同practice的数据转换逻辑
"""
from typing import Dict, Any, Callable, AsyncIterator


class PracticeRegistry:
    """Practice注册中心"""

    _transformers: Dict[str, Callable] = {}

    @classmethod
    def register(cls, practice_name: str, transformer: Callable):
        """
        注册practice的数据转换器

        Args:
            practice_name: practice名称
            transformer: 转换函数，签名为 (data_stream, data_config) -> transformed_stream
        """
        cls._transformers[practice_name] = transformer

    @classmethod
    def get_transformer(cls, practice_name: str) -> Callable:
        """
        获取practice的数据转换器

        Args:
            practice_name: practice名称

        Returns:
            转换函数，如果不存在返回None
        """
        return cls._transformers.get(practice_name)

    @classmethod
    def list_practices(cls) -> list:
        """列出所有注册的practice"""
        return list(cls._transformers.keys())

