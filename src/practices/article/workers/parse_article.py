"""
Article content 解析 worker（包含智能段落还原）。

流程：
1. 从 `Article.pdfdata` 读取 PDF 二进制。
2. 按页提取文本并做智能段落还原。
3. 写入 `Article.content`，更新 `content_*` 状态。
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import sys
from pathlib import Path

_file_path = Path(__file__).resolve()
_project_root = _file_path.parent.parent.parent.parent.parent
_src_root = _project_root / "src"
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

try:
    from ..models import Article, PDFData, init_database
    from ..utils import task_generator
    from .text_util import restore_pdf_page_text
except ImportError:
    from practices.article.models import Article, PDFData, init_database
    from practices.article.utils import task_generator
    from practices.article.workers.text_util import restore_pdf_page_text

logger = logging.getLogger(__name__)

def _normalize_content(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_pages_with_pymupdf(pdf_bytes: bytes) -> list[str]:
    import fitz

    pages: list[str] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_num in range(len(doc)):
            restored = restore_pdf_page_text(pdf_bytes, page_num)
            if restored.strip():
                pages.append(restored)
    return pages


def _extract_pages_with_pypdf(pdf_bytes: bytes) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    normalized_pages = []
    for page_text in pages:
        page_text = _normalize_content(page_text)
        if page_text:
            normalized_pages.append(page_text)
    return normalized_pages


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    errors: list[str] = []

    try:
        pages = _extract_pages_with_pymupdf(pdf_bytes)
        text = _normalize_content("\n\n".join(pages))
        if text:
            return text
        errors.append("pymupdf: 空文本")
    except Exception as exc:
        errors.append(f"pymupdf: {exc}")

    try:
        pages = _extract_pages_with_pypdf(pdf_bytes)
        text = _normalize_content("\n\n".join(pages))
        if text:
            return text
        errors.append("pypdf: 空文本")
    except Exception as exc:
        errors.append(f"pypdf: {exc}")

    raise RuntimeError("无法提取 PDF 文本，尝试后端失败: " + " | ".join(errors))


async def _fetch_pdf_bytes(article: Article) -> bytes:
    if article.pdfdata is None:
        raise ValueError("article.pdfdata 为空")

    pdf_data_doc = await article.pdfdata.fetch()
    if pdf_data_doc is None or not isinstance(pdf_data_doc, PDFData):
        raise ValueError("article.pdfdata 链接无效")
    if not pdf_data_doc.data:
        raise ValueError("article.pdfdata.data 为空")
    return pdf_data_doc.data


async def _parse_article_content(article: Article, trial_times: int) -> None:
    try:
        pdf_bytes = await _fetch_pdf_bytes(article)
        content = await asyncio.to_thread(_extract_pdf_text, pdf_bytes)
        if not content:
            raise ValueError("解析结果为空文本")

        article.content = content
        article.status = "content_parsed"
        article.trial_count = 0
        await article.save()

        logger.info("content 解析成功: article=%s, chars=%d", article.id, len(content))
    except Exception as exc:
        article.status = "content_retrying" if article.trial_count < trial_times else "content_failed"
        logger.error("content 解析失败: article=%s, error=%s", article.id, exc)
        await article.save()


async def article_parse_content_worker(connection_string: str) -> None:
    """解析 `Article.pdfdata` 到 `Article.content`。"""
    logger.info("Article parse content worker started")
    await init_database(connection_string=connection_string)

    trial_times = int(os.getenv("TRIAL_TIMES", 1))
    worker_concurrency = max(1, int(os.getenv("WORKER_CONCURRENCY", "8")))
    filter_condition = {
        "$or": [
            {"status": "pdf_parsed"},
            {"status": "content_retrying", "trial_count": {"$lt": trial_times}},
        ]
    }

    count = await Article.find(filter_condition).count()
    logger.info("待解析 content 文档数: %d (concurrency=%d)", count, worker_concurrency)

    in_flight: set[asyncio.Task] = set()

    async def _run_one(target_article: Article) -> None:
        await _parse_article_content(target_article, trial_times)

    async for article in task_generator(
        Article,
        filter=filter_condition,
        set={"status": "content_parsing"},
        inc={"trial_count": 1},
        sleep_time=int(os.getenv("SLEEP_TIME", 30)),
    ):
        task = asyncio.create_task(_run_one(article))
        in_flight.add(task)
        task.add_done_callback(in_flight.discard)

        if len(in_flight) >= worker_concurrency:
            await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)


async def pdf_parse_worker(parser: str, connection_string: str) -> None:
    """兼容旧接口：忽略 parser，仅执行 content 解析。"""
    del parser
    await article_parse_content_worker(connection_string)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    config_path = _project_root / "assets" / "database.toml"
    if not config_path.exists():
        logger.error("配置文件不存在: %s", config_path)
        sys.exit(1)

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    default_config = config.get("default", {})
    db_name = os.getenv("ARTICLE_DB_NAME") or default_config.get("db_name", "RQ")
    conn_str = default_config.get("connection_string", "")

    if "<DBNAME>" in conn_str:
        connection_string = conn_str.replace("<DBNAME>", db_name)
    else:
        connection_string = conn_str

    if not connection_string:
        logger.error("无法获取数据库连接字符串")
        sys.exit(1)

    logger.info("使用数据库: %s", db_name)
    logger.info("连接字符串: %s", connection_string)

    try:
        asyncio.run(article_parse_content_worker(connection_string))
    except KeyboardInterrupt:
        logger.info("Worker 已停止")
