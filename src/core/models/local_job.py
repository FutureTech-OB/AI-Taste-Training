"""
本地文件存储的 FineTuneJob

接口与 FineTuneJob（Beanie Document）完全相同，
但无需 MongoDB——每条任务以 JSON 文件保存在本地目录。

默认目录: .finetune/jobs/
"""
import json
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, ClassVar

from pydantic import BaseModel, Field, PrivateAttr

from ..schema.training import JobStatus


class LocalFineTuneJob(BaseModel):
    """本地文件持久化的训练任务记录（无需 MongoDB）"""

    # 类级别默认存储目录，可通过子类或直接赋值覆盖
    _default_jobs_dir: ClassVar[str] = ".finetune/jobs"

    # ── 字段（与 FineTuneJob 保持一致） ───────────────────────────────
    model: str
    base_model: str
    fine_tuned_model: Optional[str] = None
    trainer: str

    job_id: Optional[str] = None
    file_id: Optional[str] = None

    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 1e-5

    practice: str
    data_filter: Dict[str, Any] = Field(default_factory=dict)
    data_config: Dict[str, Any] = Field(default_factory=dict)

    training_samples: Optional[int] = None
    train_file_path: Optional[str] = None

    status: JobStatus = JobStatus.PENDING
    error_message: Optional[str] = None

    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    started_at: Optional[datetime.datetime] = None
    finished_at: Optional[datetime.datetime] = None

    output_dir: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # ── 私有：运行时写入，不序列化 ────────────────────────────────────
    _jobs_dir: str = PrivateAttr(default=".finetune/jobs")
    _file_path: Optional[str] = PrivateAttr(default=None)

    # ── 工厂方法：允许自定义存储目录 ──────────────────────────────────
    @classmethod
    def with_jobs_dir(cls, jobs_dir: str):
        """返回一个绑定了自定义存储目录的子类，用法同 LocalFineTuneJob"""
        class _Bound(cls):  # type: ignore[valid-type]
            _jobs_dir: str = PrivateAttr(default=jobs_dir)
        _Bound.__name__ = cls.__name__
        return _Bound

    # ── Beanie 兼容接口 ───────────────────────────────────────────────

    async def insert(self):
        """首次写入：生成文件名并持久化"""
        Path(self._jobs_dir).mkdir(parents=True, exist_ok=True)
        ts = self.created_at.strftime("%Y%m%d_%H%M%S")
        safe_model = self.model.replace("/", "_").replace(" ", "_")
        filename = f"{safe_model}_{ts}.json"
        self._file_path = str(Path(self._jobs_dir) / filename)
        self._write()

    async def save(self):
        """更新写入（若未 insert 过则自动 insert）"""
        if self._file_path is None:
            await self.insert()
        else:
            self._write()

    def update_status(self, status: JobStatus, error_message: Optional[str] = None):
        """与 FineTuneJob.update_status 完全相同"""
        self.status = status
        self.updated_at = datetime.datetime.now()

        if status == JobStatus.TRAINING and not self.started_at:
            self.started_at = datetime.datetime.now()
        elif status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED):
            self.finished_at = datetime.datetime.now()

        if error_message:
            self.error_message = error_message

    # ── 内部 ──────────────────────────────────────────────────────────

    def _write(self):
        data = self.model_dump(mode="json")
        with open(self._file_path, "w", encoding="utf-8") as f:  # type: ignore[arg-type]
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
