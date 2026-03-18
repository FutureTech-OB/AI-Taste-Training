"""
Core验证器 - 基于Message的通用验证逻辑
抽象基类，子类只需实现 converter（将数据转换为 messages）
"""
import json
import logging
import asyncio
import os
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
from tqdm import tqdm

from src.core.schema.message import Message
from src.core.schema.validation import ValOutcome
from src.core.utils.inference import inference
from src.core.utils.output_parsing import (
    parse_reasoning_and_final,
    normalize_label,
    logp_to_top1_top2,
    DEFAULT_LABELS,
    OB_RL_LABELS,
)
from src.core.dataloader.base import BaseDataLoader
from src.core.schema.filter import BaseFilter

logger = logging.getLogger(__name__)

# 这些模型若缓存中 vote_predictions 含 null，则视为缓存无效并重跑
MODELS_RETRY_IF_VOTE_NULL = ("z-ai/glm-5","google/gemini-2.5-pro","deepseek/deepseek-v3.2-speciale","anthropic/claude-opus-4.6","qwen/qwen3.5-plus-02-15")


class BaseValidator(ABC):
    """验证器抽象基类

    子类只需实现：
    1. convert_to_messages: 将数据项转换为 Message 列表
    2. extract_ground_truth: 从数据项中提取 ground truth
    3. get_item_id: 获取数据项的唯一标识符
    """

    def __init__(
        self,
        dataloader: BaseDataLoader,
        model_config: Dict[str, Any],
        provider_config: Dict[str, Any],
        entry: str = None,
        enable_logp: bool = True,
        top_logprobs: int = 5,
        max_concurrent: int = 10,
        avg_n: int = 1,
        **kwargs,
    ):
        if not model_config:
            raise ValueError("model_config 不能为空")
        if not provider_config:
            raise ValueError("provider_config 不能为空")
        if not provider_config.get("provider"):
            raise ValueError("provider_config 必须包含 provider 字段")

        model_name = model_config.get("model_name")
        if not model_name:
            raise ValueError("model_config 必须包含 model_name 字段")

        self.dataloader = dataloader
        self.model_config = model_config
        self.provider_config = provider_config
        self.model_name = model_name
        self.entry = entry
        self.enable_logp = enable_logp
        self.top_logprobs = top_logprobs
        self.max_concurrent = max_concurrent
        self.avg_n = max(1, int(avg_n or 1))
        self.kwargs = kwargs

        # 蒸馏数据存储路径（可选）：每条 request/response 追加写入此 JSONL
        distill_path = kwargs.pop("distill_path", None)
        self.distill_path: Optional[str] = distill_path
        self._distill_lock = asyncio.Lock()
        if self.distill_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.distill_path)), exist_ok=True)

    # ------------------------------------------------------------------ #
    #  子类必须实现的抽象方法
    # ------------------------------------------------------------------ #

    @abstractmethod
    def convert_to_messages(self, item: Dict[str, Any]) -> List[Message]:
        """将数据项转换为 Message 列表"""
        pass

    @abstractmethod
    def extract_ground_truth(self, item: Dict[str, Any]) -> str:
        """从数据项中提取 ground truth"""
        pass

    @abstractmethod
    def get_item_id(self, item: Dict[str, Any]) -> str:
        """获取数据项的唯一标识符"""
        pass

    # ------------------------------------------------------------------ #
    #  内部处理
    # ------------------------------------------------------------------ #

    def _get_val_outcome(self, item: Dict[str, Any]) -> Optional[Dict]:
        """从 item 中读取当前 model+entry 对应的 val_outcome（若存在）"""
        val_outcome = item.get("val_outcome") or {}
        if self.entry:
            return (val_outcome.get(self.entry) or {}).get(self.model_name)
        return val_outcome.get(self.model_name)

    def _extract_existing_result(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """如果该 item 已有缓存的验证结果，提取并返回标准格式 result dict"""
        model_outcome = self._get_val_outcome(item)
        if not model_outcome:
            return None

        # --- logp 模式缓存 ---
        if self.enable_logp:
            logp = model_outcome.get("logp") if isinstance(model_outcome, dict) else getattr(model_outcome, "logp", None)
            if not logp or not isinstance(logp, dict):
                return None

            t1, t1_lp, t2, t2_lp = logp_to_top1_top2(logp, DEFAULT_LABELS)
            if t1 is None:
                return None

            return {
                "item_id": self.get_item_id(item),
                "ground_truth": self.extract_ground_truth(item),
                "logp": {k: v for k, v in logp.items() if v is not None},
                "top1_prediction": t1,
                "top1_logprob": t1_lp,
                "top2_prediction": t2,
                "top2_logprob": t2_lp,
            }

        # --- 文本模式缓存（enable_logp=False） ---
        if not isinstance(model_outcome, dict):
            return None

        prediction = model_outcome.get("prediction")
        is_match = model_outcome.get("is_match")
        response_text = model_outcome.get("response_text")
        full_response = model_outcome.get("full_response")  # 可能为完整 response 对象（dict）或旧缓存无此键
        reasoning_meta = model_outcome.get("reasoning_meta")
        reasoning_content = model_outcome.get("reasoning_content")

        # 至少要有 prediction 或 is_match，才认为缓存有效（否则可能是旧数据/空写入）
        if prediction is None and is_match is None and response_text is None:
            return None

        vote_predictions = model_outcome.get("vote_predictions")
        # 指定模型：缓存中 vote 含 null 则视为无效，不命中缓存以触发重跑
        if self.model_name in MODELS_RETRY_IF_VOTE_NULL and vote_predictions:
            if any(p is None for p in vote_predictions):
                return None

        return {
            "item_id": self.get_item_id(item),
            "ground_truth": self.extract_ground_truth(item),
            "prediction": prediction,
            "is_match": is_match,
            "response_text": response_text,
            "full_response": full_response if full_response is not None else (response_text or ""),
            "reasoning_meta": reasoning_meta,
            "reasoning_content": reasoning_content,
            # avg voting fields (if present in cache)
            "avg_accuracy": model_outcome.get("avg_accuracy"),
            "vote_n": model_outcome.get("vote_n"),
            "vote_valid_n": model_outcome.get("vote_valid_n"),
            "vote_counts": model_outcome.get("vote_counts"),
            "vote_is_tie": model_outcome.get("vote_is_tie"),
            "vote_tied": model_outcome.get("vote_tied"),
            "vote_predictions": vote_predictions,
        }

    async def _process_item(
        self,
        item: Dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Any]:
        """处理单个数据项（在 semaphore 内调用 inference）"""
        async with semaphore:
            try:
                messages = self.convert_to_messages(item)
                if not messages:
                    return {}

                extra_params = {}
                # 关闭推理时的 provider 特有参数：
                # - openai.official(Chat Completions): 不支持 reasoning/thinking 参数，不传
                # - openrouter: 使用 reasoning.effort="none"
                # - zai: 使用 thinking.type="disabled"
                provider = (self.provider_config or {}).get("provider", "")
                if not self.kwargs.get("enable_thinking", False) and self.kwargs.get("thinking_model", False):
                    if provider == "openai.official":
                        pass  # Chat Completions 不支持；Responses API 才支持 reasoning，此处不传
                    elif provider.startswith("zai."):
                        extra_params["extra_body"] = {"thinking": {"type": "disabled"}}
                    else:
                        extra_params["extra_body"] = {"reasoning": {"effort": "none"}}

                # temperature 透传：
                # - OpenAI chat.completions：支持直接传 temperature
                # - Vertex：通过 generation_config.temperature 传递
                temperature = self.kwargs.get("temperature")
                if temperature is not None:
                    extra_params["temperature"] = float(temperature)
                    extra_params["generation_config"] = {**extra_params.get("generation_config", {}), "temperature": float(temperature)}

                max_tokens = self.kwargs.get("max_tokens")
                if max_tokens is not None:
                    extra_params["max_tokens"] = int(max_tokens)

                output: Dict[str, Any] = {
                    "item_id": self.get_item_id(item),
                    "ground_truth": self.extract_ground_truth(item),
                }

                if self.enable_logp:
                    result = await inference(
                        model_config=self.model_config,
                        provider_config=self.provider_config,
                        messages=messages,
                        enable_logp=True,
                        top_logprobs=self.top_logprobs,
                        enable_thinking=self.kwargs.get("enable_thinking", False),
                        **extra_params,
                    )
                    logp = result.get("logp")
                    if logp and isinstance(logp, dict):
                        t1, t1_lp, t2, t2_lp = logp_to_top1_top2(logp, DEFAULT_LABELS)
                        if t1 is not None:
                            output["logp"] = {k: v for k, v in logp.items() if v is not None}
                            output["top1_prediction"] = t1
                            output["top1_logprob"] = t1_lp
                            output["top2_prediction"] = t2
                            output["top2_logprob"] = t2_lp
                        else:
                            logger.warning(
                                f"[alignment_fail] logp empty after filtering item={output.get('item_id')}"
                            )
                    else:
                        rt = result.get("response_text")
                        logger.warning(
                            f"[alignment_fail] no logp returned item={output.get('item_id')} "
                            f"response={repr((rt or '')[:120])}"
                        )
                    rt = result.get("response_text")
                    output["full_response"] = result.get("full_response") or rt or ""
                    output["response_text"] = result.get("response_text")
                    await self._write_distill_record(messages, result, output)
                else:
                    # 文本模式：支持 avg_n 次采样后多数投票（例如 thinking model 的 avg8）
                    allow_tier = self.kwargs.get("allow_tier_number_format", False)
                    labels = self.kwargs.get("parsing_labels")
                    if labels is None:
                        labels = OB_RL_LABELS if allow_tier else DEFAULT_LABELS
                    gt = normalize_label(
                        output.get("ground_truth"),
                        labels=labels,
                        allow_tier_number_format=allow_tier,
                    )
                    model_outcome = self._get_val_outcome(item)
                    existing_preds: List[Optional[str]] = (
                        list(model_outcome.get("vote_predictions") or [])
                        if isinstance(model_outcome, dict) and self.model_name in MODELS_RETRY_IF_VOTE_NULL
                        else []
                    )
                    null_indices = [i for i, p in enumerate(existing_preds) if p is None] if existing_preds else []
                    n_fill = len(null_indices)

                    if n_fill > 0:
                        # 只重跑 null 数量：对 null 位做 n_fill 次推理并填回
                        preds = list(existing_preds)
                        for k, idx in enumerate(null_indices):
                            r = await inference(
                                model_config=self.model_config,
                                provider_config=self.provider_config,
                                messages=messages,
                                enable_logp=False,
                                top_logprobs=self.top_logprobs,
                                enable_thinking=self.kwargs.get("enable_thinking", False),
                                **extra_params,
                            )
                            rt = r.get("response_text")
                            _, pred = parse_reasoning_and_final(
                                rt,
                                labels=labels,
                                allow_tier_number_format=allow_tier,
                            )
                            if pred is None:
                                logger.warning(
                                    f"[alignment_fail] item={output.get('item_id')} slot={idx} "
                                    f"response={repr((rt or '')[:120])}"
                                )
                            stored_pred = pred if pred is not None else "alignment_fail"
                            preds[idx] = stored_pred
                            await self._write_distill_record(messages, r, output)
                        # 沿用缓存的 response_text / reasoning（只补全了 vote）
                        resp_texts = []
                        full_resps = []
                        reasoning_texts = []
                        reasoning_metas = []
                        match_flags = [(gt is not None) and (p == gt) and p != "alignment_fail" for p in preds]
                        n = len(preds)
                    else:
                        n = self.avg_n
                        preds = []
                        resp_texts = []
                        full_resps = []
                        reasoning_texts = []
                        reasoning_metas = []
                        match_flags = []

                        for _ in range(n):
                            r = await inference(
                                model_config=self.model_config,
                                provider_config=self.provider_config,
                                messages=messages,
                                enable_logp=False,
                                top_logprobs=self.top_logprobs,
                                enable_thinking=self.kwargs.get("enable_thinking", False),
                                **extra_params,
                            )
                            rt = r.get("response_text")
                            reasoning_content, pred = parse_reasoning_and_final(
                                rt,
                                labels=labels,
                                allow_tier_number_format=allow_tier,
                            )
                            if pred is None:
                                logger.warning(
                                    f"[alignment_fail] item={output.get('item_id')} "
                                    f"response={repr((rt or '')[:120])}"
                                )
                            stored_pred = pred if pred is not None else "alignment_fail"
                            preds.append(stored_pred)
                            resp_texts.append(rt)
                            full_resps.append(r.get("full_response"))
                            reasoning_texts.append(reasoning_content)
                            reasoning_metas.append(r.get("reasoning_meta"))
                            match_flags.append((gt is not None) and (pred == gt) if pred is not None else False)
                            await self._write_distill_record(messages, r, output)

                    # vote统计（仅统计有效 pred，排除 alignment_fail）
                    counts: Dict[str, int] = {}
                    for p in preds:
                        if p and p != "alignment_fail":
                            counts[p] = counts.get(p, 0) + 1

                    majority = None
                    tied: List[str] = []
                    is_tie = False
                    if counts:
                        max_c = max(counts.values())
                        tied = sorted([k for k, v in counts.items() if v == max_c])
                        is_tie = len(tied) > 1
                        majority = tied[0]

                    pick_idx = 0
                    if majority:
                        for i, p in enumerate(preds):
                            if p == majority:
                                pick_idx = i
                                break

                    if resp_texts:
                        chosen_text = resp_texts[pick_idx] if pick_idx < len(resp_texts) else (resp_texts[0] or "")
                        chosen_full = full_resps[pick_idx] if pick_idx < len(full_resps) else (full_resps[0] if full_resps else None)
                        chosen_reasoning = reasoning_texts[pick_idx] if pick_idx < len(reasoning_texts) else None
                        chosen_reasoning_meta = reasoning_metas[pick_idx] if pick_idx < len(reasoning_metas) else None
                    else:
                        # 仅补 null 时沿用缓存
                        chosen_text = (model_outcome or {}).get("response_text") or ""
                        chosen_full = (model_outcome or {}).get("full_response") or chosen_text
                        chosen_reasoning = (model_outcome or {}).get("reasoning_content")
                        chosen_reasoning_meta = (model_outcome or {}).get("reasoning_meta")

                    output["vote_n"] = len(preds)
                    output["vote_valid_n"] = sum(1 for p in preds if p and p != "alignment_fail")
                    output["vote_counts"] = counts
                    output["vote_is_tie"] = is_tie
                    output["vote_tied"] = tied if is_tie else None
                    output["vote_predictions"] = preds
                    output["avg_accuracy"] = (sum(match_flags) / len(match_flags)) if match_flags else None
                    output["response_text"] = chosen_text
                    output["full_response"] = chosen_full if chosen_full is not None else chosen_text
                    if chosen_reasoning_meta is not None:
                        output["reasoning_meta"] = chosen_reasoning_meta
                    if chosen_reasoning:
                        output["reasoning_content"] = chosen_reasoning
                    output["prediction"] = None
                    output["is_match"] = None

                return output
            except Exception as e:
                # 把已拿到的 responses 挂到异常上，供上层写 debug 文件
                e._resp_texts = locals().get("resp_texts", [])
                raise

    async def _write_distill_record(
        self,
        messages: List,
        inference_result: Dict[str, Any],
        output_ctx: Dict[str, Any],
    ) -> None:
        """将单次 request/response 存为独立 JSON 文件（若 distill_path 目录已设置）。
        文件名格式: {sanitized_item_id}__{counter}.json
        """
        if not self.distill_path:
            return
        record = {
            "item_id": output_ctx.get("item_id"),
            "ground_truth": output_ctx.get("ground_truth"),
            "model": self.model_name,
            "messages": [m.to_dict() if hasattr(m, "to_dict") else m for m in messages],
            "response_text": inference_result.get("response_text"),
            "reasoning_content": inference_result.get("reasoning_content"),
            "full_response": inference_result.get("full_response"),
        }
        item_id = output_ctx.get("item_id") or "unknown"
        # 清理文件名中的非法字符
        safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(item_id))
        safe_id = safe_id[:80]  # 防止路径过长
        uid = uuid.uuid4().hex[:8]
        filename = f"{safe_id}__{uid}.json"
        filepath = os.path.join(self.distill_path, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)


    async def _save_result(self, item: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """
        将结果写回 item 的 val_outcome 并持久化（支持 logp / 文本两种模式）。
        存储路径: val_outcome[entry][model]（有 entry 时）或 val_outcome[model]
        注意：full_response 不写入此处（不写回 120.jsonl），仅通过 save_results 写入 outcome/.../validation_results.jsonl。
        """
        if item.get("val_outcome") is None:
            item["val_outcome"] = {}

        if self.enable_logp:
            if not result.get("logp"):
                return False
            val = ValOutcome(logp=result["logp"]).model_dump(mode="json")
        else:
            # 文本模式：只写 ValOutcome 字段，不含 full_response（full_response 只存在于 outcome 目录的 jsonl）
            val = ValOutcome(
                response_text=result.get("response_text"),
                prediction=result.get("prediction"),
                is_match=result.get("is_match"),
                reasoning_meta=result.get("reasoning_meta"),
                reasoning_content=result.get("reasoning_content"),
                avg_accuracy=result.get("avg_accuracy"),
                vote_n=result.get("vote_n"),
                vote_valid_n=result.get("vote_valid_n"),
                vote_counts=result.get("vote_counts"),
                vote_is_tie=result.get("vote_is_tie"),
                vote_tied=result.get("vote_tied"),
                vote_predictions=result.get("vote_predictions"),
            ).model_dump(mode="json")

        if self.entry:
            entry_dict = item["val_outcome"].setdefault(self.entry, {})
            entry_dict[self.model_name] = val
        else:
            item["val_outcome"][self.model_name] = val

        return await self._save_item(item)

    async def _save_item(self, item: Dict[str, Any]) -> bool:
        """通过 dataloader 持久化数据项"""
        if hasattr(self.dataloader, "save_item"):
            return await self.dataloader.save_item(item)
        return False

    async def _process_and_save(
        self,
        item: Dict[str, Any],
        semaphore: asyncio.Semaphore,
        max_retries: int = 3,
    ) -> Tuple[Dict[str, Any], bool]:
        """处理 + 保存，返回 (result, saved)。
        API 异常最多重试 max_retries 次（指数退避）；格式错误不重试。
        """
        import traceback

        last_exc: Optional[Exception] = None
        for attempt in range(1 + max_retries):
            try:
                result = await self._process_item(item, semaphore)
                saved = False
                if result:
                    saved = await self._save_result(item, result)
                return result, saved
            except Exception as e:
                last_exc = e
                item_id = self.get_item_id(item)
                if attempt < max_retries:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(
                        f"API 异常 [{item_id}] 第 {attempt + 1} 次失败，{wait}s 后重试: {e}"
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"处理失败 [{item_id}]（已重试 {max_retries} 次）: {e}")

        # 所有重试耗尽，写 failed 文件
        item_id = self.get_item_id(item)
        if self.distill_path:
            failed_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.normpath(self.distill_path))),
                "failed",
            )
            os.makedirs(failed_dir, exist_ok=True)
            safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(item_id))[:80]
            fpath = os.path.join(failed_dir, f"{safe_id}__{uuid.uuid4().hex[:8]}.json")
            try:
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump({
                        "item_id": item_id,
                        "error": str(last_exc),
                        "traceback": traceback.format_exc(),
                        "responses": getattr(last_exc, "_resp_texts", []),
                    }, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        return {}, False

    # ------------------------------------------------------------------ #
    #  公开入口
    # ------------------------------------------------------------------ #

    async def validate(
        self,
        filter: BaseFilter,
        skip_existing: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        执行验证流程（真正并发 + 自动跳过已有结果 + 分批次持久化）

        Returns:
            所有结果列表（已缓存 + 新推理），可直接传给 calculate_metrics
        """
        logger.info(
            "Validator config: "
            f"enable_logp={self.enable_logp}, avg_n={self.avg_n}, "
            f"max_concurrent={self.max_concurrent}, "
            f"thinking_model={bool(self.kwargs.get('thinking_model', False))}, "
            f"enable_thinking={bool(self.kwargs.get('enable_thinking', False))}"
        )
        semaphore = asyncio.Semaphore(self.max_concurrent)
        existing_results: List[Dict[str, Any]] = []
        tasks: List[asyncio.Task] = []

        # Phase 1: 加载数据，分流已缓存 vs 待处理
        async for item in self.dataloader.load_stream(filter):
            item_id = self.get_item_id(item)
            if not item_id:
                continue

            if skip_existing:
                existing = self._extract_existing_result(item)
                if existing:
                    existing_results.append(existing)
                    continue

            tasks.append(asyncio.create_task(self._process_and_save(item, semaphore)))

        total_tasks = len(tasks)
        logger.info(f"已缓存: {len(existing_results)} 条, 待推理: {total_tasks} 条")

        # Phase 2: 并发推理，收集结果
        new_results: List[Dict[str, Any]] = []
        update_count = 0
        error_count = 0

        try:
            for coro in tqdm(asyncio.as_completed(tasks), total=total_tasks, desc="推理中", unit="条"):
                try:
                    result, saved = await coro
                    if result:
                        new_results.append(result)
                    else:
                        error_count += 1
                    if saved:
                        update_count += 1
                except Exception as e:
                    # 即使单个任务失败，也继续处理其他任务
                    error_count += 1
                    logger.error(f"任务执行失败: {e}")
        except Exception as e:
            logger.error(f"验证流程中断: {e}")
        finally:
            # Phase 3: 无论成功还是异常，都确保 flush 缓存到磁盘
            if hasattr(self.dataloader, "flush"):
                await self.dataloader.flush()
                logger.info("已执行最终 flush，确保所有缓存数据已保存")

        logger.info(
            f"验证完成: 新推理 {len(new_results)} 条, "
            f"保存 {update_count} 条, "
            f"失败 {error_count} 条"
        )
        return existing_results + new_results
