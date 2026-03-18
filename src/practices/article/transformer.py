"""
Article数据转换工具 - 将Article转换为训练用的messages格式
"""
import random
from typing import Dict, Any, List, AsyncIterator, Optional
from src.core.schema.message import Message, MessageRole
from .prompts import ArticlePrompts
from .utils_rank import normalize_rank


class ArticleDataTransformer:
    """Article数据转换器"""

    @staticmethod
    def to_messages(
        article: Dict[str, Any],
        entry: str,
        prompt_name: str,
        target_field: str = "rank"
    ) -> List[Message]:
        """
        将Article转换为Message对象列表

        Args:
            article: Article数据字典
            entry: 使用的entry字段（如"abstract", "full_text"）
            prompt_name: 提示词名称
            target_field: 目标字段（如"rank", "subject"）

        Returns:
            Message对象列表
        """
        # 获取提示词
        system_prompt = ArticlePrompts.get_prompt(prompt_name)
        if not system_prompt:
            system_prompt = "You are a helpful assistant."

        # 获取输入内容：仅用 entry，无 entry 不 fallback 到 title（建数据集时无 entry 的会在 transform_stream 筛掉）
        entries = article.get("entries") or {}
        user_content = (entries.get(entry) or "").strip() or ""

        # 获取目标答案（Enum 取 .name，字符串则统一为首字母大写以与 prompt 一致）
        raw_label = article.get(target_field)
        if raw_label is None:
            assistant_content = ""
        elif hasattr(raw_label, "name"):   # Enum 对象（如 Article.Rank）
            assistant_content = raw_label.name        # e.g. "Exceptional"
        else:
            s = str(raw_label or "").strip()
            # rank 别名归一化到 Exceptional/Strong/Fair/Limited
            if s and target_field == "rank":
                canon = normalize_rank(s)
                if canon:
                    assistant_content = canon.capitalize()
                else:
                    assistant_content = s
            else:
                assistant_content = s

        # 构建Message对象列表
        messages = [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
            Message(role=MessageRole.USER, content=user_content),
            Message(role=MessageRole.ASSISTANT, content=assistant_content)
        ]

        return messages

    @staticmethod
    async def transform_stream(
        data_stream: AsyncIterator[Dict[str, Any]],
        entry: str,
        prompt_name: str,
        target_field: str = "rank",
        balance_max_per_class: Optional[int] = None,
        balance_seed: int = 42,
        balance_strategy: str = "random",
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        流式转换Article数据为训练格式。

        当 balance_max_per_class 指定时，按 target_field 类别平衡：每类最多取 balance_max_per_class 条
        （不足则全取），用 balance_seed 做随机采样，再整体 shuffle 后输出。
        """
        import logging as _log
        _logger = _log.getLogger(__name__)
        total = valid = skipped_entry = skipped_label = 0
        rng = random.Random(balance_seed)

        if balance_max_per_class is None or balance_max_per_class <= 0:
            # 不 balance：原有流式逻辑
            async for article in data_stream:
                total += 1
                entries = article.get("entries") or {}
                user_content = (entries.get(entry) or "").strip()
                raw_label = article.get(target_field)
                assistant_content = raw_label.name if hasattr(raw_label, "name") else str(raw_label or "")

                if not user_content:
                    skipped_entry += 1
                    continue
                if not assistant_content:
                    skipped_label += 1
                    continue

                valid += 1
                messages = ArticleDataTransformer.to_messages(
                    article, entry, prompt_name, target_field
                )
                yield {"messages": [msg.to_dict() for msg in messages]}

            _logger.info(
                f"[transform] total={total}  valid={valid}  "
                f"skipped(no {entry})={skipped_entry}  skipped(no {target_field})={skipped_label}"
            )
            return

        # balance：先按类别收集，再每类最多采 balance_max_per_class 条
        buckets: Dict[str, List[Dict[str, Any]]] = {}

        async for article in data_stream:
            total += 1
            entries = article.get("entries") or {}
            user_content = (entries.get(entry) or "").strip()
            raw_label = article.get(target_field)
            if hasattr(raw_label, "name"):
                assistant_content = raw_label.name
            else:
                s = str(raw_label or "").strip()
                if s and target_field == "rank":
                    canon = normalize_rank(s)
                    if canon:
                        assistant_content = canon.capitalize()
                    else:
                        assistant_content = s
                else:
                    assistant_content = s

            if not user_content:
                skipped_entry += 1
                continue
            if not assistant_content:
                skipped_label += 1
                continue

            valid += 1
            messages = ArticleDataTransformer.to_messages(
                article, entry, prompt_name, target_field
            )
            item = {"messages": [msg.to_dict() for msg in messages]}
            key = assistant_content
            year = article.get("published_year") or -1
            try:
                year_int = int(year)
            except Exception:
                year_int = -1
            buckets.setdefault(key, []).append({"item": item, "year": year_int})

        # 每类最多采 balance_max_per_class 条，数量不够则全留
        sampled: List[Dict[str, Any]] = []
        for key, items in buckets.items():
            if balance_strategy == "year_desc":
                items_sorted = sorted(items, key=lambda x: x.get("year", -1), reverse=True)
                chosen = items_sorted[: balance_max_per_class]
                sampled.extend([c["item"] for c in chosen])
            else:
                if len(items) <= balance_max_per_class:
                    sampled.extend([x["item"] for x in items])
                else:
                    chosen = rng.sample(items, balance_max_per_class)
                    sampled.extend([x["item"] for x in chosen])

        rng.shuffle(sampled)
        for item in sampled:
            yield item

        _logger.info(
            f"[transform] total={total}  valid={valid}  "
            f"skipped(no {entry})={skipped_entry}  skipped(no {target_field})={skipped_label}  "
            f"balance(max_per_class={balance_max_per_class}, seed={balance_seed}) → {len(sampled)} samples"
        )
