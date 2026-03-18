"""
Dual (Pairwise) Validator — pairwise comparison experiment.

For each pair of articles at a given tier distance, the model is asked to
identify which research question has higher publication potential.
Correct = the one whose ground-truth tier is higher.
"""
import asyncio
import json
import logging
import os
import random
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from src.core.schema.message import Message, MessageRole
from src.core.utils.inference import inference
from src.core.utils.output_parsing import parse_reasoning_and_final
from src.practices.article.prompts import ArticlePrompts
from src.practices.article.utils_rank import normalize_rank

_DUAL_LABELS = ("first", "second")

logger = logging.getLogger(__name__)

# Tier order: higher index → higher quality
RANK_ORDER: Dict[str, int] = {
    "exceptional": 4,
    "strong": 3,
    "fair": 2,
    "limited": 1,
}

VALID_RANKS = set(RANK_ORDER.keys())


def _pair_cache_key(first: str, second: str) -> Tuple[str, str]:
    """缓存键：无 DOI 时用 title，统一转成字符串并 strip 以便匹配。"""
    a = str(first or "").strip()
    b = str(second or "").strip()
    return (a, b)


def _tier_distance(rank_a: str, rank_b: str) -> Optional[int]:
    """Return absolute tier distance, or None if either rank is invalid."""
    oa = RANK_ORDER.get(rank_a)
    ob = RANK_ORDER.get(rank_b)
    if oa is None or ob is None:
        return None
    return abs(oa - ob)


def _sample_pairs(
    articles: List[Dict[str, Any]],
    num_per_pair_type: int,
    seed: int,
) -> Dict[Tuple[str, str], List[Tuple[Dict, Dict]]]:
    """
    Group articles by rank; stratify by tier pair (4 tiers → C(4,2)=6 pair types).
    Randomly sample num_per_pair_type per pair type with fixed seed.

    Returns {(rank_low, rank_high): [(article_a, article_b), ...]}
    """
    def _stable_key(a: Dict[str, Any]) -> str:
        return str(a.get("doi") or a.get("title") or "")

    sorted_articles = sorted(articles, key=_stable_key)
    by_rank: Dict[str, List[Dict]] = {r: [] for r in VALID_RANKS}
    for art in sorted_articles:
        rn = normalize_rank(art.get("rank"))
        if rn in VALID_RANKS:
            by_rank[rn].append(art)

    # 4 tiers → 6 pair types: (limited,fair), (limited,strong), (limited,exceptional),
    # (fair,strong), (fair,exceptional), (strong,exceptional)
    rank_list = sorted(VALID_RANKS, key=lambda r: RANK_ORDER[r])
    by_pair_type: Dict[Tuple[str, str], List[Tuple[Dict, Dict]]] = {}
    for ra, rb in combinations(rank_list, 2):
        key = (ra, rb)
        pool_a = by_rank[ra]
        pool_b = by_rank[rb]
        by_pair_type[key] = [(a, b) for a in pool_a for b in pool_b]

    rng = random.Random(seed)
    sampled: Dict[Tuple[str, str], List[Tuple[Dict, Dict]]] = {}
    for key, pairs in by_pair_type.items():
        if not pairs:
            sampled[key] = []
            continue
        k = min(num_per_pair_type, len(pairs))
        sampled[key] = rng.sample(pairs, k)

    total = sum(len(v) for v in sampled.values())
    for key, pairs in sampled.items():
        logger.info(f"  Pair type {key[0]}-{key[1]}: {len(pairs)} pairs sampled")
    logger.info(f"Total pairs: {total}")
    return sampled


class DualValidator:
    """
    Pairwise comparison validator.

    For each pair the model sees:
        System: OB_RQCONTEXT_PROMPT_DUAL
        User:   RQ1: <entry_content_a>\\n\\nRQ2: <entry_content_b>

    Ground-truth: whichever article has the higher tier is the correct choice.
    Position of correct answer (first / second) is randomised per pair so that
    positional bias is balanced across the sample.
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        provider_config: Dict[str, Any],
        entry: str = "rq_with_context",
        prompt_name: str = "ob_rqcontext_dual",
        max_concurrent: int = 10,
        temperature: float = 0.0,
        num_per_pair_type: int = 50,
        seed: int = 42,
        thinking_model: bool = False,
        enable_thinking: bool = False,
        pair_timeout: Optional[float] = 300.0,
        **kwargs,
    ):
        if not model_config:
            raise ValueError("model_config cannot be empty")
        if not provider_config or not provider_config.get("provider"):
            raise ValueError("provider_config must contain a 'provider' field")

        self.model_config = model_config
        self.provider_config = provider_config
        self.model_name = model_config["model_name"]
        self.entry = entry
        self.prompt_name = prompt_name
        self.max_concurrent = max_concurrent
        self.temperature = temperature
        self.num_per_pair_type = num_per_pair_type
        self.seed = seed
        self.thinking_model = thinking_model
        self.enable_thinking = enable_thinking
        self.pair_timeout = pair_timeout  # 单对推理超时（秒），None 表示不超时
        self.kwargs = kwargs

    def _get_entry_text(self, article: Dict[str, Any]) -> str:
        entries = article.get("entries") or {}
        text = entries.get(self.entry) or article.get("title") or ""
        return text.strip()

    def _build_messages(
        self,
        text_first: str,
        text_second: str,
    ) -> List[Message]:
        system_prompt = ArticlePrompts.get_prompt(self.prompt_name) or ""
        user_content = f"RQ1: {text_first}\n\nRQ2: {text_second}"
        return [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
            Message(role=MessageRole.USER, content=user_content),
        ]

    async def _run_one_pair(
        self,
        pair_info: Dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Any]:
        """Run inference for a single pair and return a result dict."""
        async with semaphore:
            text_first = pair_info["text_first"]
            text_second = pair_info["text_second"]
            messages = self._build_messages(text_first, text_second)

            extra: Dict[str, Any] = {"temperature": self.temperature}

            # Thinking-model handling (mirrors BaseValidator logic)
            provider = (self.provider_config or {}).get("provider", "")
            if self.thinking_model and not self.enable_thinking:
                if provider.startswith("zai."):
                    extra["extra_body"] = {"thinking": {"type": "disabled"}}
                elif provider != "openai.official":
                    extra["extra_body"] = {"reasoning": {"effort": "none"}}

            result = await inference(
                model_config=self.model_config,
                provider_config=self.provider_config,
                messages=messages,
                enable_logp=False,
                enable_thinking=self.enable_thinking,
                **extra,
            )

            raw_text = result.get("response_text") or ""
            # Strip reasoning tags; extract final "first" / "second"
            _, prediction = parse_reasoning_and_final(raw_text, labels=_DUAL_LABELS)
            is_valid_choice = prediction in _DUAL_LABELS
            # Alignment fail: model did not answer with a valid label
            alignment_fail = not is_valid_choice
            # Treat alignment fail as incorrect for metrics
            correct = (prediction == pair_info["correct_position"]) if is_valid_choice else False

            return {
                "pair_type": pair_info["pair_type"],
                "distance": pair_info["distance"],
                "doi_first": pair_info["doi_first"],
                "doi_second": pair_info["doi_second"],
                "rank_first": pair_info["rank_first"],
                "rank_second": pair_info["rank_second"],
                "correct_position": pair_info["correct_position"],
                "prediction": prediction,
                "raw_response": raw_text.strip(),
                "is_correct": correct,
                "alignment_fail": alignment_fail,
            }

    async def validate(
        self,
        articles: List[Dict[str, Any]],
        cache_path: Optional[str] = None,
        overwrite_cache: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Run pairwise validation.

        Args:
            articles: All candidate articles (already filtered by split/subject/type).
            cache_path: Path to an existing pair_results.jsonl to load cached results
                        from. Pairs already present in the cache are skipped.
                        Pass None (or set overwrite_cache=True) to disable caching.
            overwrite_cache: If True, ignore existing cache and re-infer all pairs.

        Returns:
            List of per-pair result dicts (cached + newly inferred), preserving the
            original pair order determined by the fixed seed.
        """
        sampled = _sample_pairs(articles, self.num_per_pair_type, self.seed)

        rng = random.Random(self.seed + 1)  # separate seed for position shuffle

        pair_infos: List[Dict[str, Any]] = []
        for (rank_lo, rank_hi), pairs in sampled.items():
            dist = _tier_distance(rank_lo, rank_hi) or 0
            pair_type = f"{rank_lo}_{rank_hi}"
            for art_a, art_b in pairs:
                rank_a = normalize_rank(art_a.get("rank")) or ""
                rank_b = normalize_rank(art_b.get("rank")) or ""

                order_a = RANK_ORDER.get(rank_a, 0)
                order_b = RANK_ORDER.get(rank_b, 0)

                text_a = self._get_entry_text(art_a)
                text_b = self._get_entry_text(art_b)

                id_a = art_a.get("doi") or art_a.get("title") or ""
                id_b = art_b.get("doi") or art_b.get("title") or ""
                if rng.random() < 0.5:
                    text_first, text_second = text_a, text_b
                    rank_first, rank_second = rank_a, rank_b
                    doi_first, doi_second = id_a, id_b
                    correct_position = "first" if order_a > order_b else "second"
                else:
                    text_first, text_second = text_b, text_a
                    rank_first, rank_second = rank_b, rank_a
                    doi_first, doi_second = id_b, id_a
                    correct_position = "first" if order_b > order_a else "second"

                if not text_first or not text_second:
                    logger.warning(
                        f"Skipping pair with missing entry text "
                        f"(pair_type={pair_type}, {doi_first} vs {doi_second})"
                    )
                    continue

                pair_infos.append({
                    "pair_type": pair_type,
                    "distance": dist,
                    "text_first": text_first,
                    "text_second": text_second,
                    "rank_first": rank_first,
                    "rank_second": rank_second,
                    "doi_first": doi_first,
                    "doi_second": doi_second,
                    "correct_position": correct_position,
                })

        # ---- Cache: load existing results ----
        cached: Dict[tuple, Dict[str, Any]] = {}
        if cache_path and not overwrite_cache and os.path.isfile(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        key = _pair_cache_key(
                            rec.get("doi_first"), rec.get("doi_second")
                        )
                        cached[key] = rec
                    except json.JSONDecodeError:
                        pass
            logger.info(f"Cache loaded: {len(cached)} pairs from {cache_path}")

        # ---- Split: cached vs pending ----
        cached_results: List[Dict[str, Any]] = []
        pending_infos: List[Dict[str, Any]] = []
        for info in pair_infos:
            key = _pair_cache_key(info["doi_first"], info["doi_second"])
            if key in cached:
                cached_results.append(cached[key])
            else:
                pending_infos.append(info)

        logger.info(
            f"Total pairs: {len(pair_infos)} | "
            f"Cached (skip): {len(cached_results)} | "
            f"Pending (infer): {len(pending_infos)}"
        )

        # ---- Infer pending pairs（增量写入 cache_path，避免卡住/中断后重跑全部）----
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [
            asyncio.create_task(self._run_one_pair(info, semaphore))
            for info in pending_infos
        ]

        new_results: List[Dict[str, Any]] = []
        errors = 0
        # 有 cache 路径时追加写入，每完成一对就 flush，下次运行可复用
        append_file = None
        if cache_path and pending_infos:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                append_file = open(cache_path, "a", encoding="utf-8")
            except OSError as e:
                logger.warning("无法打开 cache 文件进行增量写入: %s", e)

        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Pairwise inference",
            unit="pair",
        ):
            try:
                if self.pair_timeout is not None:
                    res = await asyncio.wait_for(coro, timeout=self.pair_timeout)
                else:
                    res = await coro
                new_results.append(res)
                if append_file is not None:
                    append_file.write(json.dumps(res, ensure_ascii=False) + "\n")
                    append_file.flush()
            except asyncio.TimeoutError:
                logger.error("Pair inference timeout (%.0fs), will retry on next run", self.pair_timeout or 0)
                errors += 1
            except Exception as e:
                logger.error(f"Pair inference failed: {e}")
                errors += 1
            finally:
                pass

        if append_file is not None:
            try:
                append_file.close()
            except OSError:
                pass

        logger.info(
            f"Done. {len(new_results)} new results, "
            f"{len(cached_results)} from cache, "
            f"{errors} errors."
        )
        return cached_results + new_results


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per pair_type (6 tier-pair categories) and overall accuracy.

    Also reports alignment failures: responses not in {"first", "second"}.
    """
    from collections import defaultdict

    by_pair_type: Dict[str, List] = defaultdict(list)
    align_fail_by_type: Dict[str, int] = defaultdict(int)
    total_align_fail = 0
    for r in results:
        pt = r.get("pair_type")
        if not pt:
            continue
        if r.get("alignment_fail"):
            total_align_fail += 1
            align_fail_by_type[pt] += 1
        if r.get("is_correct") is not None:
            by_pair_type[pt].append(r["is_correct"])

    per_pair_type: Dict[str, Any] = {}
    total_correct = 0
    total_valid = 0

    for pt in sorted(by_pair_type.keys()):
        flags = by_pair_type[pt]
        n = len(flags)
        correct = sum(flags)
        acc = correct / n if n else 0.0
        per_pair_type[pt] = {
            "n": n,
            "correct": correct,
            "accuracy": acc,
            "alignment_fail": align_fail_by_type.get(pt, 0),
        }
        total_correct += correct
        total_valid += n

    overall_acc = total_correct / total_valid if total_valid else 0.0

    if per_pair_type:
        weighted = sum(v["accuracy"] for v in per_pair_type.values()) / len(per_pair_type)
    else:
        weighted = 0.0

    return {
        "total": total_valid,
        "total_correct": total_correct,
        "overall_accuracy": overall_acc,
        "weighted_accuracy": weighted,
        "alignment_fail_total": total_align_fail,
        "per_pair_type": per_pair_type,
    }
