"""
OpenAI SFT 训练器
"""
import json
import asyncio
import os
import re
from pathlib import Path
from typing import AsyncIterator, Dict, Any, Optional
from openai import AsyncOpenAI
import httpx
from ..trainer import BaseSFTTrainer
from ...schema.training import TrainingConfig
from ...models.finetune import FineTuneJob
from ...dataloader.converter import DataConverter


# 轮询状态时单次请求超时（秒），避免代理/网络慢导致立即失败
OPENAI_POLL_TIMEOUT = 120.0
# 超时/连接错误时重试次数与间隔（秒）
OPENAI_POLL_RETRIES = 5
OPENAI_POLL_RETRY_DELAYS = (10, 20, 30, 60, 90)


def _safe_filename(s: str) -> str:
    """替换 Windows 非法字符（如 ft: 模型 id 中的冒号），避免 OSError [Errno 22]。"""
    if not s or not isinstance(s, str):
        return "model"
    s = re.sub(r'[<>:"/\\|?*]', "_", s.strip())
    return s or "model"


class OpenAITrainer(BaseSFTTrainer):
    """OpenAI Fine-tuning训练器"""

    def __init__(self, api_key: str, job_class=None, timeout: float = OPENAI_POLL_TIMEOUT):
        super().__init__("openai", job_class=job_class)
        self.client = AsyncOpenAI(api_key=api_key, timeout=httpx.Timeout(timeout))

    async def prepare_training_data(
        self,
        data_stream: AsyncIterator[Dict[str, Any]],
        config: TrainingConfig,
        job: FineTuneJob
    ) -> str:
        """准备训练数据 - 转换为OpenAI格式的JSONL"""
        safe_model = _safe_filename(job.model)
        output_path = f"{config.output_dir}/{safe_model}_train.jsonl"

        async def transform_stream():
            async for item in data_stream:
                if "messages" in item:
                    yield {"messages": item["messages"]}
                else:
                    yield item

        await DataConverter.to_jsonl(transform_stream(), output_path)

        sample_count = 0
        import aiofiles
        async with aiofiles.open(output_path, 'r', encoding='utf-8') as f:
            async for _ in f:
                sample_count += 1

        job.training_samples = sample_count
        import logging as _logging
        _logging.getLogger(__name__).info(
            f"[prepare] {sample_count} samples → {output_path}"
        )
        return output_path

    async def create_training_job(
        self,
        train_file_path: str,
        config: TrainingConfig,
        job: FineTuneJob
    ) -> str:
        """创建OpenAI训练任务"""
        with open(train_file_path, 'rb') as f:
            file_response = await self.client.files.create(
                file=f,
                purpose='fine-tune'
            )

        job.file_id = file_response.id

        create_kwargs = {
            "training_file": file_response.id,
            "model": config.model,
            "hyperparameters": {
                "n_epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate_multiplier": config.learning_rate,
            },
        }
        if suffix := config.platform_config.get("suffix"):
            create_kwargs["suffix"] = suffix
        ft_job = await self.client.fine_tuning.jobs.create(**create_kwargs)

        return ft_job.id

    async def _fetch_all_events(self, job_id: str, limit: int = 100):
        """分页拉取全部事件"""
        all_events = []
        after = None
        while True:
            kwargs = {"limit": limit}
            if after is not None:
                kwargs["after"] = after
            page = await self.client.fine_tuning.jobs.list_events(job_id, **kwargs)
            if not page.data:
                break
            all_events.extend(page.data)
            if len(page.data) < limit:
                break
            after = page.data[-1].id
        return all_events

    async def _build_job_all_md(self, job_id: str) -> str:
        """拉取 status + events + checkpoints 并格式化为一份 Markdown（与 jobs all 一致）"""
        job = await self.client.fine_tuning.jobs.retrieve(job_id)
        status_data = {
            "id": job.id,
            "status": job.status,
            "model": job.model,
            "fine_tuned_model": job.fine_tuned_model,
            "trained_tokens": job.trained_tokens,
            "created_at": job.created_at,
            "estimated_finish": job.estimated_finish,
            "finished_at": job.finished_at,
            "error": job.error.model_dump() if job.error else None,
        }
        events = await self._fetch_all_events(job_id)
        events_md_lines = [
            "| 时间 | 级别 | 消息 |",
            "|------|------|------|",
        ]
        for e in reversed(events):
            msg = (e.message or "").replace("|", "\\|").replace("\n", " ")
            events_md_lines.append(f"| {e.created_at} | {e.level or ''} | {msg} |")
        checkpoints = await self.client.fine_tuning.jobs.checkpoints.list(job_id)
        cp_lines = []
        for cp in (checkpoints.data or []):
            cp_lines.append(json.dumps({
                "id": cp.id,
                "step_number": cp.step_number,
                "fine_tuned_model_checkpoint": cp.fine_tuned_model_checkpoint,
                "created_at": cp.created_at,
                "metrics": cp.metrics.model_dump() if cp.metrics else None,
            }, indent=2, default=str))
        return (
            f"# Fine-tuning Job: {job_id}\n\n"
            "## STATUS\n\n```json\n"
            + json.dumps(status_data, indent=2, default=str)
            + "\n```\n\n## EVENTS\n\n"
            + ("共 {} 条事件（时间正序）\n\n".format(len(events)) + "\n".join(events_md_lines) if events else "（暂无事件）")
            + "\n\n## CHECKPOINTS\n\n"
            + ("\n\n".join(cp_lines) if cp_lines else "（暂无检查点）")
        )

    async def wait_for_completion(
        self, job: FineTuneJob, config: Optional[TrainingConfig] = None
    ) -> Dict[str, Any]:
        """等待训练完成（每 30s 轮询一次，打印状态；若提供 config 则定期将完整 job 信息写入 log 文件）"""
        import logging as _logging
        import time
        from openai import APITimeoutError
        import httpx

        _logger = _logging.getLogger(__name__)
        poll_interval = 30
        start = time.time()
        job_log_path = None
        if config:
            job_log_path = config.platform_config.get("job_log_file") or os.path.join(
                config.output_dir, f"ft_job_{job.job_id}.md"
            )

        while True:
            ft_job = None
            for attempt in range(OPENAI_POLL_RETRIES + 1):
                try:
                    if attempt > 0:
                        delay = OPENAI_POLL_RETRY_DELAYS[attempt - 1]
                        await asyncio.sleep(delay)
                        _logger.warning(
                            "[4/4] 轮询超时/连接失败，重试 %d/%d (等待 %ds 后)",
                            attempt,
                            OPENAI_POLL_RETRIES + 1,
                            delay,
                        )
                    ft_job = await self.client.fine_tuning.jobs.retrieve(job.job_id)
                    break
                except (APITimeoutError, httpx.TimeoutException, httpx.ConnectError) as e:
                    if attempt >= OPENAI_POLL_RETRIES:
                        raise
            if ft_job is None:
                raise RuntimeError("轮询获取 job 状态失败")

            elapsed = int(time.time() - start)
            elapsed_str = f"{elapsed // 60}m{elapsed % 60:02d}s"

            if job_log_path:
                try:
                    md = await self._build_job_all_md(job.job_id)
                    Path(job_log_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(job_log_path).write_text(md, encoding="utf-8")
                except Exception as e:
                    _logger.debug("写入 job log 失败: %s", e)

            if ft_job.status == "succeeded":
                _logger.info(f"[4/4] succeeded  elapsed={elapsed_str}  tokens={ft_job.trained_tokens}")
                return {
                    "fine_tuned_model": ft_job.fine_tuned_model,
                    "trained_tokens": ft_job.trained_tokens,
                }
            elif ft_job.status in ["failed", "cancelled"]:
                raise Exception(f"Training {ft_job.status}: {ft_job.error}")
            else:
                _logger.info(f"[4/4] status={ft_job.status}  elapsed={elapsed_str}  job_id={job.job_id}")

            await asyncio.sleep(poll_interval)

    async def cancel(self, job: FineTuneJob) -> bool:
        """取消训练任务"""
        try:
            await self.client.fine_tuning.jobs.cancel(job.job_id)
            return True
        except Exception:
            return False

    async def get_status(self, job: FineTuneJob) -> Dict[str, Any]:
        """获取训练状态"""
        ft_job = await self.client.fine_tuning.jobs.retrieve(job.job_id)

        return {
            "job_id": job.job_id,
            "status": ft_job.status,
            "fine_tuned_model": ft_job.fine_tuned_model,
            "trained_tokens": ft_job.trained_tokens,
            "created_at": ft_job.created_at,
            "finished_at": ft_job.finished_at,
        }
