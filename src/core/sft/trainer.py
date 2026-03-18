"""
抽象SFT训练器基类
"""
import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Any, Optional, Callable
from ..schema.training import TrainingConfig, JobStatus
from ..models.finetune import FineTuneJob
import datetime

_logger = logging.getLogger(__name__)


class BaseSFTTrainer(ABC):
    """SFT训练器抽象基类"""

    def __init__(self, trainer_name: str, job_class=None):
        """
        Args:
            trainer_name: 训练器名称（openai/google/trl/verl）
            job_class: 训练任务记录类，默认 FineTuneJob（需 MongoDB）；
                       无 DB 时可传入 LocalFineTuneJob
        """
        self.trainer_name = trainer_name
        self._job_class = job_class if job_class is not None else FineTuneJob

    @abstractmethod
    async def prepare_training_data(
        self,
        data_stream: AsyncIterator[Dict[str, Any]],
        config: TrainingConfig,
        job: FineTuneJob
    ) -> str:
        """
        准备训练数据

        Args:
            data_stream: 数据流
            config: 训练配置
            job: 训练任务记录

        Returns:
            训练文件路径
        """
        pass

    @abstractmethod
    async def create_training_job(
        self,
        train_file_path: str,
        config: TrainingConfig,
        job: FineTuneJob
    ) -> str:
        """
        创建训练任务

        Args:
            train_file_path: 训练文件路径
            config: 训练配置
            job: 训练任务记录

        Returns:
            平台返回的job_id
        """
        pass

    @abstractmethod
    async def wait_for_completion(
        self,
        job: FineTuneJob,
        config: Optional[TrainingConfig] = None,
    ) -> Dict[str, Any]:
        """
        等待训练完成

        Args:
            job: 训练任务记录
            config: 训练配置（可选，OpenAI 等用于写 job 状态 log）

        Returns:
            训练结果
        """
        pass

    async def prepare_eval_data(
        self,
        eval_stream: AsyncIterator[Dict[str, Any]],
        config: TrainingConfig,
        job: FineTuneJob,
    ) -> None:
        """准备验证数据（可选实现，子类覆盖）"""

    async def train(
        self,
        data_stream: AsyncIterator[Dict[str, Any]],
        config: TrainingConfig,
        practice: str,
        data_filter: Dict[str, Any],
        data_config: Dict[str, Any],
        transform_func = None,
        eval_stream: Optional[AsyncIterator[Dict[str, Any]]] = None,
        eval_transform_func: Optional[Callable] = None,
    ) -> FineTuneJob:
        """
        完整训练流程（模板方法）

        Args:
            data_stream: 数据流
            config: 训练配置
            practice: practice名称
            data_filter: 数据过滤条件
            data_config: 数据配置
            transform_func: 可选的数据转换函数，用于将原始数据转换为messages格式
                          签名: async (data_stream, data_config) -> transformed_stream

        Returns:
            训练任务记录
        """
        if transform_func:
            data_stream = transform_func(data_stream, data_config)
        if eval_stream is not None and eval_transform_func is not None:
            eval_stream = eval_transform_func(eval_stream, data_config)

        # 1. 创建训练任务记录
        job = self._job_class(
            model=config.model_name or f"{config.model}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            base_model=config.model,
            trainer=self.trainer_name,
            practice=practice,
            batch_size=config.batch_size,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            data_filter=data_filter,
            data_config=data_config,
            output_dir=config.output_dir,
            status=JobStatus.PREPARING
        )
        await job.insert()

        try:
            # 2. 准备训练数据
            _logger.info("[1/4] 准备训练数据中 ...")
            job.update_status(JobStatus.PREPARING)
            await job.save()

            train_file_path = await self.prepare_training_data(data_stream, config, job)
            job.train_file_path = train_file_path
            await job.save()

            if eval_stream is not None:
                await self.prepare_eval_data(eval_stream, config, job)

            # 3. 上传并创建训练任务
            _logger.info(f"[2/4] 上传训练文件: {train_file_path}  ({job.training_samples} samples)")
            job.update_status(JobStatus.UPLOADING)
            await job.save()

            platform_job_id = await self.create_training_job(train_file_path, config, job)
            job.job_id = platform_job_id
            job.update_status(JobStatus.TRAINING)
            await job.save()
            _logger.info(f"[3/4] 训练任务已提交: job_id={platform_job_id}")

            # 4. 等待完成
            _logger.info("[4/4] 等待训练完成 ...")
            result = await self.wait_for_completion(job, config)

            # 5. 更新结果
            job.fine_tuned_model = result.get("fine_tuned_model")
            job.update_status(JobStatus.SUCCEEDED)
            await job.save()
            _logger.info(f"[done] fine_tuned_model={job.fine_tuned_model}")

            return job

        except Exception as e:
            job.update_status(JobStatus.FAILED, error_message=str(e))
            await job.save()
            raise

    async def cancel(self, job: FineTuneJob) -> bool:
        """
        取消训练任务（可选实现）

        Args:
            job: 训练任务记录

        Returns:
            是否成功取消
        """
        raise NotImplementedError("Cancel not supported by this trainer")

    async def get_status(self, job: FineTuneJob) -> Dict[str, Any]:
        """
        获取训练状态（可选实现）

        Args:
            job: 训练任务记录

        Returns:
            状态信息
        """
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "progress": None,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        }
