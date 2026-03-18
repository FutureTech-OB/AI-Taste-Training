"""
注册Article Practice到训练框架
"""
from .. import PracticeRegistry
from .transformer import ArticleDataTransformer


def register_article_practice():
    """注册Article practice的数据转换器"""

    async def article_transformer(data_stream, data_config):
        """
        Article数据转换器

        Args:
            data_stream: Article数据流
            data_config: 数据配置，包含：
                - entry: 使用的entry字段（如"abstract", "full_text"）
                - prompt: 提示词名称（如"rank_classifier"）
                - target_field: 目标字段（如"rank", "subject"）

        Returns:
            转换后的数据流（包含messages）
        """
        entry = data_config.get("entry", "abstract")
        prompt_name = data_config.get("prompt", "rank_classifier")
        target_field = data_config.get("target_field", "rank")

        return ArticleDataTransformer.transform_stream(
            data_stream,
            entry=entry,
            prompt_name=prompt_name,
            target_field=target_field
        )

    # 注册到全局注册中心
    PracticeRegistry.register("article", article_transformer)
