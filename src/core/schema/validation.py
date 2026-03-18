"""
验证相关的数据模式
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ValidationStatus(str, Enum):
    """验证任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ValidationMetrics(BaseModel):
    """验证指标"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    per_class_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    custom_metrics: Dict[str, Any] = Field(default_factory=dict)


class ValidationConfig(BaseModel):
    """验证配置"""
    model: str = Field(..., description="要验证的模型")
    batch_size: int = Field(default=10, ge=1)
    metrics: List[str] = Field(
        default=["accuracy", "f1"],
        description="要计算的指标"
    )
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None, ge=1)


class ValOutcome(BaseModel):
    """验证结果模型，写回源数据（如 120.jsonl）的 val_outcome。
    注意：完整 API response（full_response）不存于此，仅写入 outcome 目录下 validation_results.jsonl。
    """
    logp: Optional[Dict[str, Optional[float]]] = None
    # 文本模式（enable_logp=False）下保存的字段，用于缓存与复算指标
    response_text: Optional[str] = None
    prediction: Optional[str] = None
    is_match: Optional[bool] = None
    reasoning_content: Optional[str] = None
    reasoning_meta: Optional[Dict[str, Any]] = None
    error: Optional[str] = None  # 错误信息（如果处理失败）
    # ---- avg voting (text-mode, e.g. avg8) ----
    avg_accuracy: Optional[float] = None          # 每条样本 avg_n 次的平均命中率（0~1）
    vote_n: Optional[int] = None                 # 总采样次数
    vote_valid_n: Optional[int] = None           # 成功解析出标签的次数
    vote_counts: Optional[Dict[str, int]] = None # 各标签票数（仅统计 valid）
    vote_is_tie: Optional[bool] = None           # 是否出现并列第一
    vote_tied: Optional[List[str]] = None        # 并列第一的标签列表（若 tie）
    vote_predictions: Optional[List[Optional[str]]] = None  # 每次采样解析出的标签（可能为 None）


class QA(BaseModel):
    """问答对模型"""
    query: str
    answer: str
    id: str
