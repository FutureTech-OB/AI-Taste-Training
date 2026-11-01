"""
DeepSpeed SFT 训练器
"""
import asyncio
import logging
import os
import subprocess
import uuid
from pathlib import Path
from typing import AsyncIterator, Dict, Any, Optional

from ..trainer import BaseSFTTrainer
from ...schema.training import TrainingConfig
from ...models.finetune import FineTuneJob
from ...dataloader.converter import DataConverter
from .config import DeepSpeedSFTConfig

_logger = logging.getLogger(__name__)

_TRAIN_SCRIPT = str(Path(__file__).parent / "train.py")


class DeepSpeedTrainer(BaseSFTTrainer):
    """DeepSpeed + TRL SFT 本地训练器"""

    def __init__(self, ds_config: DeepSpeedSFTConfig, job_class=None):
        super().__init__("deepspeed", job_class=job_class)
        self.ds_config = ds_config
        self._process: Optional[subprocess.Popen] = None

    async def prepare_eval_data(
        self,
        eval_stream: AsyncIterator[Dict[str, Any]],
        config: TrainingConfig,
        job: FineTuneJob,
    ) -> None:
        """将验证数据写入 JSONL 并注入 ds_config.eval_file"""
        eval_path = str(Path(config.output_dir) / f"{job.model}_eval.jsonl")

        async def _normalize(stream: AsyncIterator[Dict[str, Any]]):
            async for item in stream:
                if "messages" in item:
                    yield {"messages": item["messages"]}
                elif "text" in item:
                    yield {"text": item["text"]}
                else:
                    yield item

        await DataConverter.to_jsonl(_normalize(eval_stream), eval_path)
        count = sum(1 for _ in open(eval_path, encoding="utf-8"))
        self.ds_config.eval_file = eval_path
        _logger.info("[prepare_eval] %d samples → %s", count, eval_path)

    async def prepare_training_data(
        self,
        data_stream: AsyncIterator[Dict[str, Any]],
        config: TrainingConfig,
        job: FineTuneJob,
    ) -> str:
        """将数据流写入本地 JSONL 文件"""
        output_path = str(Path(config.output_dir) / f"{job.model}_train.jsonl")

        async def _normalize(stream: AsyncIterator[Dict[str, Any]]):
            async for item in stream:
                if "messages" in item:
                    yield {"messages": item["messages"]}
                elif "text" in item:
                    yield {"text": item["text"]}
                else:
                    yield item

        await DataConverter.to_jsonl(_normalize(data_stream), output_path)

        sample_count = sum(1 for _ in open(output_path, encoding="utf-8"))
        job.training_samples = sample_count
        _logger.info("[prepare] %d samples → %s", sample_count, output_path)
        return output_path

    async def create_training_job(
        self,
        train_file_path: str,
        config: TrainingConfig,
        job: FineTuneJob,
    ) -> str:
        """构建并启动 deepspeed 训练子进程，返回进程 PID"""
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        log_basename = f"train_{uuid.uuid4().hex[:8]}.log"
        log_path = str(Path(config.output_dir) / log_basename)
        (Path(config.output_dir) / ".train_log").write_text(log_basename, encoding="utf-8")

        cmd = self._build_command(train_file_path, config)
        _logger.info("[create] cmd: %s", " ".join(cmd))
        _logger.info("[create] stdout/stderr → %s", log_path)

        with open(log_path, "w", encoding="utf-8") as log_f:
            process = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)

        self._process = process
        _logger.info("[create] PID=%d", process.pid)
        return str(process.pid)

    async def wait_for_completion(
        self, job: FineTuneJob, config: Optional[TrainingConfig] = None
    ) -> Dict[str, Any]:
        """阻塞等待训练子进程结束"""
        if self._process is None:
            raise RuntimeError("训练进程未启动，请先调用 create_training_job")

        returncode = await asyncio.to_thread(self._process.wait)
        if returncode != 0:
            log_path = _resolve_train_log_path(job.output_dir)
            log_tail = _read_tail(log_path)
            raise RuntimeError(
                f"DeepSpeed 训练失败 (exit code {returncode})\n{log_tail}"
            )

        _logger.info("[done] training succeeded, output: %s", job.output_dir)
        return {"fine_tuned_model": job.output_dir or ""}

    async def cancel(self, job: FineTuneJob) -> bool:
        """终止训练子进程"""
        if self._process is None:
            return False
        try:
            self._process.terminate()
            await asyncio.sleep(5)
            if self._process.poll() is None:
                self._process.kill()
            return True
        except Exception as exc:
            _logger.warning("cancel failed: %s", exc)
            return False

    async def get_status(self, job: FineTuneJob) -> Dict[str, Any]:
        base = await super().get_status(job)
        if self._process is not None:
            base["pid"] = self._process.pid
            base["returncode"] = self._process.poll()
        return base

    def _build_command(self, train_file_path: str, config: TrainingConfig) -> list[str]:
        ds = self.ds_config
        # 不传 --num_gpus，让 DeepSpeed 按 CUDA_VISIBLE_DEVICES 自动检测卡数
        cmd: list[str] = ["deepspeed", "--master_port", str(ds.master_port), _TRAIN_SCRIPT]

        cmd += ["--data_file", train_file_path]
        if ds.eval_file:
            cmd += ["--eval_file", ds.eval_file]
            cmd += ["--eval_strategy", ds.eval_strategy]
        if ds.dataset_text_field != "text":
            cmd += ["--dataset_text_field", ds.dataset_text_field]

        cmd += ["--model_path", ds.model_path]
        if ds.attn_implementation:
            cmd += ["--attn_implementation", ds.attn_implementation]

        cmd += ["--output_dir", config.output_dir]
        cmd += ["--epochs", str(config.epochs)]
        cmd += ["--lr", str(config.learning_rate)]
        cmd += ["--per_device_batch_size", str(ds.per_device_batch_size)]
        cmd += ["--batch_size", str(config.batch_size)]
        cmd += ["--max_length", str(ds.max_length)]
        cmd += ["--pad_to_multiple_of", str(ds.pad_to_multiple_of)]
        if not ds.completion_only_loss:
            cmd.append("--no-completion-only-loss")
        cmd += ["--warmup_ratio", str(ds.warmup_ratio)]
        cmd += ["--lr_scheduler_type", ds.lr_scheduler_type]
        cmd += ["--save_steps", str(ds.save_steps)]
        cmd += ["--save_strategy", ds.save_strategy]
        if ds.save_only_model:
            cmd.append("--save_only_model")
        cmd += ["--logging_steps", str(ds.logging_steps)]

        if not ds.bf16:
            cmd.append("--no-bf16")
        if ds.gradient_checkpointing:
            cmd.append("--gradient_checkpointing")
        if ds.packing:
            cmd.append("--packing")
        if ds.deepspeed_config:
            cmd += ["--deepspeed", ds.deepspeed_config]

        if ds.use_peft:
            cmd.append("--use_peft")
            cmd += ["--lora_r", str(ds.lora_r)]
            cmd += ["--lora_alpha", str(ds.lora_alpha)]
            cmd += ["--lora_dropout", str(ds.lora_dropout)]
            if ds.lora_target_modules:
                cmd += ["--lora_target_modules", *ds.lora_target_modules]

        return cmd


def _detect_gpus() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return max(1, len(result.stdout.strip().splitlines()))
    except Exception:
        pass
    return 1


def _resolve_train_log_path(output_dir: Optional[str]) -> Optional[str]:
    """解析本次运行的 train log 路径（train_<uuid>.log 或兼容旧版 train.log）。"""
    if not output_dir:
        return None
    p = Path(output_dir)
    marker = p / ".train_log"
    if marker.exists():
        try:
            basename = marker.read_text(encoding="utf-8").strip()
            return str(p / basename)
        except Exception:
            pass
    return str(p / "train.log")


def _read_tail(log_path: Optional[str], lines: int = 30) -> str:
    if not log_path or not Path(log_path).exists():
        return ""
    try:
        with open(log_path, encoding="utf-8") as f:
            return "".join(f.readlines()[-lines:])
    except Exception:
        return ""
