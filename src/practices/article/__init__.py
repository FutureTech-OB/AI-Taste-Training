"""
Article Practice - 研究文章数据管理
"""
from .models import Article
from .schema import ArticleFilter
from .loader import ArticleLoader
from .prompts import ArticlePrompts
from .transformer import ArticleDataTransformer
from .register import register_article_practice
from .workers import article_fill_pdfdata_worker
from .workers import article_parse_type_worker
from .ob import (
    ARTICLE_EXTRACTION_PROMPT,
    OB_RQCONTEXT_PROMPT,
    OB_RQCONTEXT_PROMPT_DUAL,
    OB_RQCONTEXT_PROMPT_JOURNAL,
    OB_RQCONTEXT_PROMPT_SIMPLE,
    SOCIAL_SCIENCE_RQCONTEXT_PROMPT,
)

# 自动注册
register_article_practice()

# 注册 OB 领域的 prompts 到 ArticlePrompts
ArticlePrompts.add_prompt("ob_rqcontext", OB_RQCONTEXT_PROMPT)
ArticlePrompts.add_prompt("ob_rqcontext_dual", OB_RQCONTEXT_PROMPT_DUAL)
ArticlePrompts.add_prompt("ob_rqcontext_simple", OB_RQCONTEXT_PROMPT_SIMPLE)
ArticlePrompts.add_prompt("ob_rqcontext_journal", OB_RQCONTEXT_PROMPT_JOURNAL)
ArticlePrompts.add_prompt("social_science_rqcontext", SOCIAL_SCIENCE_RQCONTEXT_PROMPT)

__all__ = [
    "Article",
    "ValOutcome",
    "ArticleFilter",
    "ArticleLoader",
    "ArticlePrompts",
    "ArticleDataTransformer",
    "ARTICLE_EXTRACTION_PROMPT",
    "article_fill_pdfdata_worker",
    "article_parse_type_worker",
]
