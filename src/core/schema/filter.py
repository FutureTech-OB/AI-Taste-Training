"""
通用过滤器定义
"""
from typing import List, Any, Dict
from pydantic import BaseModel, Field
from enum import Enum


class FilterOperator(str, Enum):
    """过滤操作符"""
    EQ = "eq"               # 等于
    NE = "ne"               # 不等于
    IN = "in"               # 在列表中
    NIN = "nin"             # 不在列表中
    GT = "gt"               # 大于
    GTE = "gte"             # 大于等于
    LT = "lt"               # 小于
    LTE = "lte"             # 小于等于
    CONTAINS = "contains"   # 包含
    REGEX = "regex"         # 正则匹配


class FilterField(BaseModel):
    """单个过滤字段"""
    name: str = Field(..., description="字段名")
    operator: FilterOperator = Field(..., description="操作符")
    value: Any = Field(..., description="过滤值")


class BaseFilter(BaseModel):
    """通用过滤器基类"""
    filters: List[FilterField] = Field(default_factory=list)

    def add_filter(self, name: str, operator: FilterOperator, value: Any):
        """添加过滤条件"""
        self.filters.append(FilterField(name=name, operator=operator, value=value))

    def to_mongo_query(self) -> Dict:
        """转换为MongoDB查询"""
        query = {}
        for f in self.filters:
            if f.operator == FilterOperator.EQ:
                query[f.name] = f.value
            elif f.operator == FilterOperator.NE:
                query[f.name] = {"$ne": f.value}
            elif f.operator == FilterOperator.IN:
                query[f.name] = {"$in": f.value}
            elif f.operator == FilterOperator.NIN:
                query[f.name] = {"$nin": f.value}
            elif f.operator == FilterOperator.GT:
                query[f.name] = {"$gt": f.value}
            elif f.operator == FilterOperator.GTE:
                query[f.name] = {"$gte": f.value}
            elif f.operator == FilterOperator.LT:
                query[f.name] = {"$lt": f.value}
            elif f.operator == FilterOperator.LTE:
                query[f.name] = {"$lte": f.value}
            elif f.operator == FilterOperator.CONTAINS:
                query[f.name] = {"$regex": f.value, "$options": "i"}
            elif f.operator == FilterOperator.REGEX:
                query[f.name] = {"$regex": f.value}
        return query
