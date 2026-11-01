"""
Article数据模型
"""
import datetime
from pydantic import Field
from beanie import Document, Indexed, Link
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from src.core.schema.validation import ValOutcome
    from .pdf import PDFData


class Article(Document):
    """研究文章模型"""

    title: Optional[Indexed(str)] = None
    authors: Optional[List[str]] = None
    published_year: Optional[int] = None
    journal: Optional[str] = None
    issn: Optional[str] = None
    open_access_path: Optional[str] = None
    doi: Optional[Indexed(str)] = None
    type: Optional[str] = None
    fwci: Optional[float] = None
    bibtex: Optional[str] = None

    class Rank(str, Enum):
        """论文质量评级

        Top: 高影响力期刊，方法论严谨，结论可靠，被广泛引用
        Good: 较好的期刊，研究设计合理，有一定贡献
        Fair: 一般期刊，研究基本规范，但存在一些局限
        Poor: 质量较差，方法或结论存在明显问题
        """
        Exceptional = "exceptional"
        Strong = "strong"
        Fair = "fair"
        Limited = "limited"
        Others = "others"
        No_Match = "no_match"

    rank: Optional[Rank] = None
    split: Optional[str] = None
    subject: Optional[str] = None
    val_outcome: Optional[Dict[str, Dict[str, Any]]] = None  # entry -> model -> ValOutcome

    # 关联数据
    pdfdata: Optional[Link["PDFData"]] = None  # 外链 PDF 二进制数据
    content: Optional[str] = None  # PDF文本内容
    metadata: Optional[Dict[str, Any]] = None
    entries: Dict[str, Any] = Field(default_factory=dict)

    # 状态
    status: str = "created"
    trial_count: int = 0
    abort: Optional[bool] = False  # 中止标记

    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.now)

    class Settings:
        name = "Article"
        indexes = [
            "title",
            "published_year",
            "journal",
            "doi",
            "rank",
            "split",
            "subject",
        ]

    class Status(Enum):
        CREATED = "created"
        PDF_PARSING = "pdf_parsing"
        PDF_PARSED = "pdf_parsed"
        PDF_FAILED = "pdf_failed"
        PDF_RETRYING = "pdf_retrying"
        DOI_PARSING = "doi_parsing"
        DOI_PARSED = "doi_parsed"
        DOI_FAILED = "doi_failed"
        DOI_RETRYING = "doi_retrying"
        ENTRIES_PARSING = "entries_parsing"
        ENTRIES_PARSED = "entries_parsed"
        ENTRIES_FAILED = "entries_failed"
        ENTRIES_RETRYING = "entries_retrying"
        METADATA_SEARCHING = "metadata_searching"
        METADATA_MAINTAINED = "metadata_maintained"
        METADATA_FAILED = "metadata_failed"
        METADATA_RETRYING = "metadata_retrying"
        MAINTAINING = "maintaining"
        ERROR = "error"


from .pdf import PDFData

Article.model_rebuild(_types_namespace={"PDFData": PDFData})
