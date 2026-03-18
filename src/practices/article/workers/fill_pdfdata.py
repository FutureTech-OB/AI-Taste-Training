"""
根据 `assets/OBarticles` 的真实结构回填 Article.pdfdata。

数据结构约定：
1. `assets/OBarticles/wep.xlsx` 是主索引。
2. `assets/OBarticles/wep/*.pdf` 的文件名是 `metadata.baseid`。
3. `assets/OBarticles/AMJ/*.pdf` 的文件名是论文标题（补充来源）。

匹配顺序：
1. 优先使用 Article.metadata.baseid 命中 `wep`。
2. 其次用 doi/rawid/title 命中 xlsx 行，再取该行 baseid 命中 `wep`。
3. 最后按标题命中 `AMJ`。
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Optional

_file_path = Path(__file__).resolve()
_project_root = _file_path.parent.parent.parent.parent.parent
_src_root = _project_root / "src"
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from ..models import Article, PDFData, init_database
    from ..utils import task_generator
except ImportError:
    from practices.article.models import Article, PDFData, init_database
    from practices.article.utils import task_generator

logger = logging.getLogger(__name__)


def _parse_subjects_env() -> list[str]:
    raw = os.getenv("ARTICLE_SUBJECTS", "ECONOMICS,SOCIOLOGY")
    subjects = [item.strip().upper() for item in raw.split(",") if item.strip()]
    return subjects or ["ECONOMICS", "SOCIOLOGY"]


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _to_plain(value: Any) -> Any:
    if _is_empty(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _normalize_text(value: Any) -> str:
    if _is_empty(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    return re.sub(r"[^a-z0-9]+", "", text)


def _normalize_identifier(value: Any) -> str:
    if _is_empty(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = text.replace(" ", "")
    if text.startswith("/doi/"):
        text = text[5:]
    return text


def _nested_set(data: dict[str, Any], path: list[str], value: Any) -> None:
    current = data
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def _deep_merge_missing(target: Optional[dict[str, Any]], source: dict[str, Any]) -> dict[str, Any]:
    if not target:
        return dict(source)
    for key, value in source.items():
        if isinstance(value, dict):
            existing = target.get(key)
            if isinstance(existing, dict):
                _deep_merge_missing(existing, value)
            elif key not in target or target[key] in (None, "", {}):
                target[key] = dict(value)
        elif key not in target or target[key] in (None, ""):
            target[key] = value
    return target


def _parse_excel_row(row: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for column, value in row.items():
        column = str(column)
        plain = _to_plain(value)
        if plain is None:
            continue
        if column.startswith("metadata."):
            _nested_set(metadata, column.split(".")[1:], plain)

    return {
        "_id": _to_plain(row.get("_id")),
        "title": _to_plain(row.get("title")),
        "subject": _to_plain(row.get("subject")),
        "doi": _normalize_identifier(row.get("doi")),
        "open_access_path": _to_plain(row.get("open_access_path")),
        "pdf_name": _to_plain(row.get("PDF名称")),
        "rawid": _normalize_identifier(metadata.get("rawid")),
        "metadata_id": _normalize_identifier(metadata.get("id")),
        "baseid": _normalize_identifier(metadata.get("baseid")),
        "has_fulltext": bool(_to_plain(row.get("has_fulltext"))),
        "metadata": metadata,
    }


def _load_excel_indexes(excel_path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    if pd is None:
        raise ImportError("需要安装 pandas 和 openpyxl")

    df = pd.read_excel(excel_path)
    by_baseid: dict[str, dict[str, Any]] = {}
    by_doi: dict[str, dict[str, Any]] = {}
    by_rawid: dict[str, dict[str, Any]] = {}
    by_metadata_id: dict[str, dict[str, Any]] = {}
    by_pdf_name: dict[str, dict[str, Any]] = {}
    by_open_access_path: dict[str, dict[str, Any]] = {}
    by_title: dict[str, dict[str, Any]] = {}

    for _, row in df.iterrows():
        record = _parse_excel_row(row)

        if record["baseid"] and record["baseid"] not in by_baseid:
            by_baseid[record["baseid"]] = record
        if record["doi"] and record["doi"] not in by_doi:
            by_doi[record["doi"]] = record
        if record["rawid"] and record["rawid"] not in by_rawid:
            by_rawid[record["rawid"]] = record
        if record["metadata_id"] and record["metadata_id"] not in by_metadata_id:
            by_metadata_id[record["metadata_id"]] = record
        pdf_name_key = _normalize_identifier(record["pdf_name"])
        if pdf_name_key and pdf_name_key not in by_pdf_name:
            by_pdf_name[pdf_name_key] = record
        open_access_key = _normalize_identifier(Path(str(record["open_access_path"] or "")).name)
        if open_access_key and open_access_key not in by_open_access_path:
            by_open_access_path[open_access_key] = record

        title_key = _normalize_text(record["title"])
        if title_key and title_key not in by_title:
            by_title[title_key] = record

    return {
        "by_baseid": by_baseid,
        "by_doi": by_doi,
        "by_rawid": by_rawid,
        "by_metadata_id": by_metadata_id,
        "by_pdf_name": by_pdf_name,
        "by_open_access_path": by_open_access_path,
        "by_title": by_title,
    }


def _build_pdf_indexes(pdf_root: Path) -> dict[str, dict[str, Path]]:
    by_name: dict[str, Path] = {}
    by_stem: dict[str, Path] = {}
    by_subject_name: dict[str, Path] = {}
    by_subject_stem: dict[str, Path] = {}

    if not pdf_root.exists():
        return {
            "by_name": by_name,
            "by_stem": by_stem,
            "by_subject_name": by_subject_name,
            "by_subject_stem": by_subject_stem,
        }

    for pdf_path in pdf_root.rglob("*.pdf"):
        subject = pdf_path.parent.name.upper()
        name_key = _normalize_identifier(pdf_path.name)
        stem_key = _normalize_identifier(pdf_path.stem)
        if name_key and name_key not in by_name:
            by_name[name_key] = pdf_path
        if stem_key and stem_key not in by_stem:
            by_stem[stem_key] = pdf_path
        if subject and name_key:
            by_subject_name.setdefault(f"{subject}:{name_key}", pdf_path)
        if subject and stem_key:
            by_subject_stem.setdefault(f"{subject}:{stem_key}", pdf_path)

    return {
        "by_name": by_name,
        "by_stem": by_stem,
        "by_subject_name": by_subject_name,
        "by_subject_stem": by_subject_stem,
    }


def _locate_row(article: Article, excel_indexes: dict[str, dict[str, dict[str, Any]]]) -> tuple[Optional[dict[str, Any]], str]:
    metadata = article.metadata or {}

    open_access_key = _normalize_identifier(Path(str(article.open_access_path or "")).name)
    if open_access_key and open_access_key in excel_indexes["by_open_access_path"]:
        return excel_indexes["by_open_access_path"][open_access_key], "article.open_access_path"

    baseid = _normalize_identifier(metadata.get("baseid"))
    if baseid and baseid in excel_indexes["by_pdf_name"]:
        return excel_indexes["by_pdf_name"][baseid], "article.metadata.baseid->pdf_name"

    baseid = _normalize_identifier(metadata.get("baseid"))
    if baseid and baseid in excel_indexes["by_baseid"]:
        return excel_indexes["by_baseid"][baseid], "article.metadata.baseid"

    doi = _normalize_identifier(article.doi)
    if doi and doi in excel_indexes["by_doi"]:
        return excel_indexes["by_doi"][doi], "article.doi"

    rawid = _normalize_identifier(metadata.get("rawid"))
    if rawid and rawid in excel_indexes["by_rawid"]:
        return excel_indexes["by_rawid"][rawid], "article.metadata.rawid"

    metadata_id = _normalize_identifier(metadata.get("id"))
    if metadata_id and metadata_id in excel_indexes["by_metadata_id"]:
        return excel_indexes["by_metadata_id"][metadata_id], "article.metadata.id"

    title_key = _normalize_text(article.title)
    if title_key and title_key in excel_indexes["by_title"]:
        return excel_indexes["by_title"][title_key], "article.title"

    return None, "none"


def _match_pdf_path(
    article: Article,
    excel_row: Optional[dict[str, Any]],
    pdf_indexes: dict[str, dict[str, Path]],
) -> tuple[Optional[Path], str]:
    metadata = article.metadata or {}
    subject = str(article.subject or (excel_row or {}).get("subject") or "").upper().strip()

    def _find_pdf(key: str, *, by_name: bool, source: str) -> tuple[Optional[Path], str]:
        if not key:
            return None, "none"
        subject_map = pdf_indexes["by_subject_name" if by_name else "by_subject_stem"]
        global_map = pdf_indexes["by_name" if by_name else "by_stem"]
        if subject:
            subject_key = f"{subject}:{key}"
            if subject_key in subject_map:
                return subject_map[subject_key], f"{source}:subject"
        if key in global_map:
            return global_map[key], source
        return None, "none"

    open_access_key = _normalize_identifier(Path(str(article.open_access_path or "")).name)
    pdf_path, pdf_source = _find_pdf(open_access_key, by_name=True, source="article.open_access_path")
    if pdf_path is not None:
        return pdf_path, pdf_source

    row_open_access_key = _normalize_identifier(Path(str((excel_row or {}).get("open_access_path") or "")).name)
    pdf_path, pdf_source = _find_pdf(row_open_access_key, by_name=True, source="xlsx.open_access_path")
    if pdf_path is not None:
        return pdf_path, pdf_source

    row_pdf_name = _normalize_identifier((excel_row or {}).get("pdf_name"))
    pdf_path, pdf_source = _find_pdf(row_pdf_name, by_name=True, source="xlsx.pdf_name")
    if pdf_path is not None:
        return pdf_path, pdf_source

    article_baseid = _normalize_identifier(metadata.get("baseid"))
    pdf_path, pdf_source = _find_pdf(article_baseid, by_name=False, source="article.metadata.baseid")
    if pdf_path is not None:
        return pdf_path, pdf_source

    if excel_row:
        row_baseid = _normalize_identifier(excel_row.get("baseid"))
        pdf_path, pdf_source = _find_pdf(row_baseid, by_name=False, source="xlsx.baseid")
        if pdf_path is not None:
            return pdf_path, pdf_source

    return None, "none"


async def _upsert_pdfdata(article: Article, pdf_bytes: bytes) -> None:
    pdf_data_doc = None
    if article.pdfdata is not None:
        try:
            pdf_data_doc = await article.pdfdata.fetch()
        except Exception as exc:
            logger.warning("读取 article.pdfdata 失败，将重建链接: %s (%s)", article.id, exc)

    if pdf_data_doc is None:
        pdf_data_doc = PDFData(data=pdf_bytes)
        await pdf_data_doc.insert()
    else:
        pdf_data_doc.data = pdf_bytes
        await pdf_data_doc.save()

    article.pdfdata = pdf_data_doc


async def _fill_pdfdata_for_article(
    article: Article,
    excel_indexes: dict[str, dict[str, dict[str, Any]]],
    pdf_indexes: dict[str, dict[str, Path]],
    trial_times: int,
) -> None:
    try:
        excel_row, row_source = _locate_row(article, excel_indexes)
        pdf_path, pdf_source = _match_pdf_path(article, excel_row, pdf_indexes)

        if pdf_path is None:
            raise FileNotFoundError(
                f"未找到 PDF: article={article.id}, title={article.title}, row_source={row_source}, pdf_source={pdf_source}"
            )

        pdf_bytes = pdf_path.read_bytes()
        if not pdf_bytes:
            raise ValueError(f"PDF 文件为空: {pdf_path}")

        await _upsert_pdfdata(article, pdf_bytes)
        article.open_access_path = str(pdf_path)

        if excel_row:
            article.metadata = _deep_merge_missing(article.metadata, excel_row.get("metadata", {}))

        article.status = "pdf_parsed"
        article.trial_count = 0
        await article.save()

        logger.info(
            "pdfdata 回填成功: article=%s, file=%s, row_source=%s, pdf_source=%s",
            article.id,
            pdf_path.name,
            row_source,
            pdf_source,
        )
    except Exception as exc:
        article.status = "pdf_retrying" if article.trial_count < trial_times else "pdf_failed"
        logger.error("pdfdata 回填失败: article=%s, error=%s", article.id, exc)
        await article.save()


async def article_fill_pdfdata_worker(
    connection_string: str,
    excel_path: Optional[str] = None,
    pdf_root: Optional[str] = None,
    min_f1: Optional[float] = None,
) -> None:
    del min_f1
    logger.info("Article fill pdfdata worker started")
    await init_database(connection_string=connection_string)

    assets_root = Path(pdf_root) if pdf_root else (_project_root / "assets" / "RIarticles")
    resolved_excel_path = Path(excel_path) if excel_path else next(assets_root.glob("*.xlsx"), None)
    if resolved_excel_path is None:
        raise FileNotFoundError(f"未找到 Excel 索引文件: {assets_root}")

    if not resolved_excel_path.exists():
        raise FileNotFoundError(f"xlsx 不存在: {resolved_excel_path}")
    if not assets_root.exists():
        raise FileNotFoundError(f"未找到 PDF 目录: {assets_root}")

    excel_indexes = _load_excel_indexes(resolved_excel_path)
    pdf_indexes = _build_pdf_indexes(assets_root)

    logger.info(
        "索引加载完成: xlsx(baseid=%d,doi=%d,rawid=%d,metadata_id=%d,pdf_name=%d,title=%d), pdf(name=%d,stem=%d)",
        len(excel_indexes["by_baseid"]),
        len(excel_indexes["by_doi"]),
        len(excel_indexes["by_rawid"]),
        len(excel_indexes["by_metadata_id"]),
        len(excel_indexes["by_pdf_name"]),
        len(excel_indexes["by_title"]),
        len(pdf_indexes["by_name"]),
        len(pdf_indexes["by_stem"]),
    )

    trial_times = int(os.getenv("TRIAL_TIMES", 1))
    subjects = _parse_subjects_env()
    filter_condition = {
        "$and": [
            {"type": "study"},
            {"subject": {"$in": subjects}},
            {"$or": [{"abort": False}, {"abort": None}, {"abort": {"$exists": False}}]},
            {"entries.rq_with_context": None},
            {
                "$or": [
                    {"status": "created"},
                    {"status": "parsing_type"},
                    {"status": "pdf_retrying", "trial_count": {"$lt": trial_times}},
                ]
            },
        ]
    }

    count = await Article.find(filter_condition).count()
    logger.info("待回填 pdfdata 文档数: %d", count)

    async for article in task_generator(
        Article,
        filter=filter_condition,
        set={"status": "pdf_parsing"},
        inc={"trial_count": 1},
        sleep_time=int(os.getenv("SLEEP_TIME", 30)),
    ):
        await _fill_pdfdata_for_article(article, excel_indexes, pdf_indexes, trial_times)


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
        asyncio.run(article_fill_pdfdata_worker(connection_string))
    except KeyboardInterrupt:
        logger.info("Worker 已停止")
