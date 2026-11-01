"""
通用微调任务模型
"""
import datetime
from beanie import Document, Indexed
from typing import Optional, Dict, Any
from pydantic import Field
from ..schema.training import JobStatus


class FineTuneJob(Document):
    """微调任务记录（通用模型，适用于所有practice）"""

    # 基本信息
    model: Indexed(str)                      # 训练任务名称/标识
    base_model: str                          # 基础模型名称
    fine_tuned_model: Optional[str] = None   # 训练后的模型名称
    trainer: str                             # 训练器类型（openai/google/trl/verl）

    # 平台特定信息
    job_id: Optional[str] = None             # 平台返回的job ID
    file_id: Optional[str] = None            # 训练文件 ID（API训练用）

    # 训练参数
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 1e-5

    # 数据配置（通用）
    practice: str                            # practice名称（如"article"）
    data_filter: Dict[str, Any] = Field(default_factory=dict)  # 数据过滤条件
    data_config: Dict[str, Any] = Field(default_factory=dict)  # 数据配置（entry, prompt等）

    # 训练数据统计
    training_samples: Optional[int] = None   # 训练样本数量
    train_file_path: Optional[str] = None    # 训练文件路径（JSONL/Parquet）

    # 状态和结果
    status: JobStatus = JobStatus.PENDING
    error_message: Optional[str] = None

    # 时间戳
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    started_at: Optional[datetime.datetime] = None
    finished_at: Optional[datetime.datetime] = None

    # 额外信息
    output_dir: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Settings:
        name = "finetune_jobs"
        indexes = [
            "model",
            "base_model",
            "trainer",
            "practice",
            "status",
            "job_id",
            "created_at",
        ]

    def to_summary_dict(self) -> Dict[str, Any]:
        """返回摘要信息"""
        return {
            "model": self.model,
            "base_model": self.base_model,
            "fine_tuned_model": self.fine_tuned_model,
            "trainer": self.trainer,
            "practice": self.practice,
            "status": self.status.value,
            "job_id": self.job_id,
            "training_params": {
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
            },
            "data_config": self.data_config,
            "data_filter": self.data_filter,
            "training_samples": self.training_samples,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "error_message": self.error_message,
        }

    def update_status(self, status: JobStatus, error_message: Optional[str] = None):
        """更新状态"""
        self.status = status
        self.updated_at = datetime.datetime.now()

        if status == JobStatus.TRAINING and not self.started_at:
            self.started_at = datetime.datetime.now()
        elif status in [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED]:
            self.finished_at = datetime.datetime.now()

        if error_message:
            self.error_message = error_message
