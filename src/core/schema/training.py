"""
训练相关的数据模式
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    """训练任务状态"""
    PENDING = "pending"          # 待开始
    PREPARING = "preparing"      # 准备数据中
    UPLOADING = "uploading"      # 上传中
    TRAINING = "training"        # 训练中
    SUCCEEDED = "succeeded"      # 成功
    FAILED = "failed"            # 失败
    CANCELLED = "cancelled"      # 已取消


class TrainingProgress(BaseModel):
    """训练进度信息"""
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    loss: Optional[float] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class TrainingConfig(BaseModel):
    """通用训练配置"""
    # 基础配置
    model: str = Field(..., description="基础模型名称")

    # 训练超参数
    batch_size: int = Field(default=16, ge=1)
    epochs: int = Field(default=3, ge=1)
    learning_rate: float = Field(default=1e-5, gt=0)

    # 输出配置
    output_dir: str = Field(default="./outputs")
    model_name: Optional[str] = Field(default=None, description="训练后的模型名称")

    # 平台特定配置（可选）
    platform_config: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "model": "gpt-4o-mini-2024-07-18",
                "batch_size": 16,
                "epochs": 3,
                "learning_rate": 1e-5,
                "output_dir": "./outputs",
            }
        }
