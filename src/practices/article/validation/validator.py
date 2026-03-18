"""
Article 验证器 - 继承 BaseValidator，实现 Article 特定的 converter
"""
import logging
from typing import List, Dict, Any

from src.core.schema.message import Message, MessageRole
from src.core.validation.validator import BaseValidator
from src.core.dataloader.base import BaseDataLoader
from src.core.utils.output_parsing import OB_RL_LABELS
from src.practices.article.prompts import ArticlePrompts

logger = logging.getLogger(__name__)


class ArticleValidator(BaseValidator):
    """Article 验证器

    只实现 3 个 converter 方法，其余逻辑全部复用 BaseValidator。
    """

    def __init__(
        self,
        dataloader: BaseDataLoader,
        model_config: Dict[str, Any],
        provider_config: Dict[str, Any],
        entry: str,
        prompt_name: str,
        enable_logp: bool = True,
        top_logprobs: int = 5,
        max_concurrent: int = 10,
        **kwargs,
    ):
        # OB_RL prompt: allow "Tier 1".."Tier 5" in parsing and use 5-tier labels.
        if prompt_name == "ob_rl":
            kwargs = dict(kwargs)
            kwargs["allow_tier_number_format"] = True
            kwargs["parsing_labels"] = OB_RL_LABELS
        super().__init__(
            dataloader=dataloader,
            model_config=model_config,
            provider_config=provider_config,
            entry=entry,
            enable_logp=enable_logp,
            top_logprobs=top_logprobs,
            max_concurrent=max_concurrent,
            **kwargs,
        )
        self.prompt_name = prompt_name

    # ---------- converter ---------- #

    def convert_to_messages(self, item: Dict[str, Any]) -> List[Message]:
        system_prompt = ArticlePrompts.get_prompt(self.prompt_name)
        if not system_prompt:
            logger.warning(f"未找到 prompt '{self.prompt_name}'，使用默认 prompt")
            system_prompt = "You are a helpful assistant."

        entries = item.get("entries") or {}
        entry_content = entries.get(self.entry)
        if not entry_content:
            entry_content = item.get("title", "")
        if not entry_content:
            return []

        return [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
            Message(role=MessageRole.USER, content=entry_content),
        ]

    def extract_ground_truth(self, item: Dict[str, Any]) -> str:
        rank = item.get("rank")
        if rank is None:
            return ""
        if isinstance(rank, str):
            return rank
        if hasattr(rank, "value"):
            return str(rank.value)
        return str(rank)

    def get_item_id(self, item: Dict[str, Any]) -> str:
        return item.get("doi") or item.get("title") or ""

