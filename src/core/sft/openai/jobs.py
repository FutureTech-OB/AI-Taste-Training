"""
OpenAI Fine-tuning Job 管理 CLI

用法:
    python -m src.core.sft.openai.jobs <action> <job_id> [model_name] [events_output.md]

action 可选值:
    status       - 查看任务状态
    events       - 查看训练事件日志
    checkpoints  - 查看保存的检查点
    cancel       - 取消任务
    all          - 显示上述全部信息

可选参数 events_output.md:
    当 action 为 events 或 all 时，可传入第 4 个参数为 .md 文件路径，
    将完整事件列表（不截断）写入该文件。
"""
import sys
import asyncio
import json
import logging
from pathlib import Path

from src.core.utils.config import ConfigLoader

logger = logging.getLogger(__name__)

CONFIG_PATH = "assets/model.toml"
DEFAULT_PROVIDER = "openai.official"


def _get_client(config_path: str = CONFIG_PATH, provider: str = DEFAULT_PROVIDER):
    from openai import AsyncOpenAI

    provider_config = ConfigLoader.get_provider_config_with_env_fallback(config_path, provider)
    api_key = provider_config.get("api_key")
    if not api_key:
        raise ValueError(
            f"无法从 {config_path} 获取 provider={provider} 的 API key，"
            "请检查 assets/model.toml 或设置 OPENAI_API_KEY 环境变量"
        )
    base_url = provider_config.get("base_url")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)


async def do_status(client, job_id: str):
    """查看任务状态"""
    job = await client.fine_tuning.jobs.retrieve(job_id)
    data = {
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
    print(json.dumps(data, indent=2, default=str))


async def _fetch_all_events(client, job_id: str, limit: int = 100):
    """分页拉取全部事件（不截断）"""
    all_events = []
    after = None
    while True:
        kwargs = {"limit": limit}
        if after is not None:
            kwargs["after"] = after
        page = await client.fine_tuning.jobs.list_events(job_id, **kwargs)
        if not page.data:
            break
        all_events.extend(page.data)
        if len(page.data) < limit:
            break
        after = page.data[-1].id
    return all_events


def _events_to_md(events: list, job_id: str) -> str:
    """将事件列表转为 Markdown 文本（API 返回新→旧，此处转为时间正序）"""
    lines = [
        f"# Fine-tuning Job Events: {job_id}",
        "",
        f"共 {len(events)} 条事件（按时间正序）",
        "",
        "| 时间 | 级别 | 消息 |",
        "|------|------|------|",
    ]
    for e in reversed(events):
        # 消息中可能含 | 或换行，做简单转义
        msg = (e.message or "").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {e.created_at} | {e.level or ''} | {msg} |")
    return "\n".join(lines)


async def do_events(client, job_id: str, events_file: str | Path | None = None):
    """查看训练事件日志。若指定 events_file 则完整写入该文件（不截断）。"""
    if events_file:
        all_events = await _fetch_all_events(client, job_id)
        if not all_events:
            print("（暂无事件）")
            return
        path = Path(events_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_events_to_md(all_events, job_id), encoding="utf-8")
        print(f"（完整 {len(all_events)} 条事件已写入 {path}）")
        # 终端仅打印最后几条摘要
        for e in reversed(all_events[-20:]):
            print(f"[{e.created_at}] {e.level.upper():8s} {e.message}")
        return
    events = await client.fine_tuning.jobs.list_events(job_id, limit=50)
    if not events.data:
        print("（暂无事件）")
        return
    for e in reversed(events.data):
        print(f"[{e.created_at}] {e.level.upper():8s} {e.message}")


async def do_checkpoints(client, job_id: str):
    """查看保存的检查点"""
    checkpoints = await client.fine_tuning.jobs.checkpoints.list(job_id)
    if not checkpoints.data:
        print("（暂无检查点）")
        return
    for cp in checkpoints.data:
        data = {
            "id": cp.id,
            "step_number": cp.step_number,
            "fine_tuned_model_checkpoint": cp.fine_tuned_model_checkpoint,
            "created_at": cp.created_at,
            "metrics": cp.metrics.model_dump() if cp.metrics else None,
        }
        print(json.dumps(data, indent=2, default=str))
        print()


async def do_cancel(client, job_id: str):
    """取消任务"""
    result = await client.fine_tuning.jobs.cancel(job_id)
    print(f"已取消: id={result.id}, status={result.status}")


async def do_all(client, job_id: str, events_file: str | Path | None = None):
    """显示任务状态 + 事件 + 检查点。若指定 events_file 则完整事件写入该文件。"""
    print("=" * 50)
    print("STATUS")
    print("=" * 50)
    await do_status(client, job_id)

    print("\n" + "=" * 50)
    print("EVENTS")
    print("=" * 50)
    await do_events(client, job_id, events_file=events_file)

    print("\n" + "=" * 50)
    print("CHECKPOINTS")
    print("=" * 50)
    await do_checkpoints(client, job_id)


ACTIONS = {
    "status": do_status,
    "events": do_events,
    "checkpoints": do_checkpoints,
    "cancel": do_cancel,
    "all": do_all,
}


async def main():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if len(sys.argv) < 3:
        print(__doc__)
        print(f"可用 action: {list(ACTIONS.keys())}")
        sys.exit(1)

    action = sys.argv[1]
    job_id = sys.argv[2]
    events_file = (sys.argv[4].strip() or None) if len(sys.argv) > 4 else None

    if action not in ACTIONS:
        print(f"未知 action: {action!r}，可用: {list(ACTIONS.keys())}")
        sys.exit(1)

    client = _get_client()
    if action == "all":
        await do_all(client, job_id, events_file=events_file)
    elif action == "events":
        await do_events(client, job_id, events_file=events_file)
    else:
        await ACTIONS[action](client, job_id)


if __name__ == "__main__":
    asyncio.run(main())
