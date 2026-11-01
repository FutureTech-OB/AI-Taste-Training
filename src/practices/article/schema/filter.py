"""
Article过滤器定义
"""
from typing import Optional, List
from pydantic import Field, model_validator
from src.core.schema.filter import BaseFilter, FilterOperator


class ArticleFilter(BaseFilter):
    """Article特定的过滤器（提供便捷接口）"""

    split: Optional[str] = Field(default=None, description="数据集划分（train/val/test）")
    subjects: Optional[List[str]] = Field(default=None, description="学科列表")
    years: Optional[List[int]] = Field(default=None, description="年份列表")
    ranks: Optional[List[str]] = Field(default=None, description="等级列表")
    types: Optional[List[str]] = Field(default=None, description="文章类型列表（如 study、review 等）")

    @model_validator(mode='after')
    def build_filters(self):
        """自动构建filters列表"""
        # 检查并添加filters（避免重复添加）
        existing_filter_names = {f.name for f in self.filters}
        
        if self.split and "split" not in existing_filter_names:
            self.add_filter("split", FilterOperator.EQ, self.split)

        if self.subjects and "subject" not in existing_filter_names:
            self.add_filter("subject", FilterOperator.IN, self.subjects)

        if self.years and "published_year" not in existing_filter_names:
            self.add_filter("published_year", FilterOperator.IN, self.years)

        if self.ranks and "rank" not in existing_filter_names:
            self.add_filter("rank", FilterOperator.IN, self.ranks)

        if self.types and "type" not in existing_filter_names:
            self.add_filter("type", FilterOperator.IN, self.types)

        return self

