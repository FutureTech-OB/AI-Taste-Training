"""
SFT 训练前置初始化工具

提供 init_job_tracking()，统一处理：
  - 有 DB 配置 → 初始化 Beanie，返回 FineTuneJob
  - 无 DB 配置 → 跳过 MongoDB，返回 LocalFineTuneJob（写入本地文件）

各 practice 的 __main__.py 只需传入自己特有的 document_models，
无需关心 job_class 的选择逻辑。
"""
import logging
from typing import Optional

_logger = logging.getLogger(__name__)


async def init_job_tracking(
    db_name: Optional[str] = None,
    connection_string: Optional[str] = None,
    extra_document_models: Optional[list] = None,
    jobs_dir: str = ".finetune/jobs",
) -> type:
    """
    初始化任务追踪并返回对应的 job_class。

    Args:
        db_name:               MongoDB 数据库名称；None 表示不使用 MongoDB
        connection_string:     MongoDB 连接字符串；优先于 db_name 拼接的默认地址
        extra_document_models: practice 自有的 Beanie Document 类列表
                               （如 [Article, PDFData, PDFDocument]）
        jobs_dir:              无 DB 时 LocalFineTuneJob 的存储目录

    Returns:
        FineTuneJob（Beanie）或 LocalFineTuneJob（本地文件）
    """
    from ..models.finetune import FineTuneJob

    if db_name or connection_string:
        from motor.motor_asyncio import AsyncIOMotorClient  # type: ignore[import]
        from beanie import init_beanie  # type: ignore[import]

        conn = connection_string or f"mongodb://localhost:27017/{db_name}"
        document_models = [FineTuneJob] + (extra_document_models or [])
        await init_beanie(
            database=AsyncIOMotorClient(conn)[db_name or "RQ"],
            document_models=document_models,
        )
        _logger.info("任务追踪: MongoDB (%s)", db_name or conn)
        return FineTuneJob
    else:
        from ..models.local_job import LocalFineTuneJob

        job_class = (
            LocalFineTuneJob.with_jobs_dir(jobs_dir)
            if jobs_dir != ".finetune/jobs"
            else LocalFineTuneJob
        )
        _logger.info("任务追踪: 本地文件 (%s)", jobs_dir)
        return job_class
