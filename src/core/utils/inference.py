"""
统一推理接口 - 支持 OpenAI 和 Vertex AI
根据 provider 自动选择对应的客户端
"""
import logging
import threading
from typing import List, Dict, Optional, Any, Tuple
from openai import AsyncOpenAI

try:
    import vertexai
    from google.oauth2 import service_account
    from vertexai.generative_models import GenerativeModel, GenerationConfig
    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False
    vertexai = None
    service_account = None
    GenerativeModel = None
    GenerationConfig = None

from src.core.schema.message import Message, MessageRole

logger = logging.getLogger(__name__)

# ── OpenAI 缓存 ──
_openai_client_cache: Dict[Tuple[str, Optional[str]], AsyncOpenAI] = {}

# ── Vertex AI 缓存 ──
_vertex_init_lock = threading.Lock()
_vertex_init_done: set = set()                        # (project_id, location, cred_path) 已初始化
_credentials_cache: Dict[str, Any] = {}               # credentials_path -> Credentials
_vertex_model_lock = threading.Lock()
_vertex_model_cache: Dict[Tuple[str, Optional[str]], Any] = {}  # (model_resource, sys_instruction) -> GenerativeModel

# ── 通用常量 ──
# logprobs 关键词集合（两个路径共用）
_LOGP_KEYWORDS = frozenset({"exceptional", "strong", "fair", "limited"})

# 仅 Vertex AI 使用的 kwargs 键，不应传递给 OpenAI API
_VERTEX_ONLY_KWARGS = frozenset({"generation_config", "timeout"})

# OpenAI 内部控制用 kwargs（项目自用），不应直接透传给 OpenAI API
_OPENAI_INTERNAL_KWARGS = frozenset({"enable_thinking", "reasoning_effort", "text_verbosity"})


def _openai_response_to_dict(obj: Any) -> Optional[Dict[str, Any]]:
    """将 OpenAI 返回的 response 对象转为可写入 JSON/JSONL 的 dict（完整 response，非仅 text）。"""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return None


def _extract_responses_output_text(resp: Any) -> str:
    """
    Best-effort extraction of output text from OpenAI Responses API response.
    Compatible with openai python SDK v1.x variations.
    """
    if resp is None:
        return ""

    # Newer SDKs: resp.output_text is a property or method
    ot = getattr(resp, "output_text", None)
    if callable(ot):
        try:
            val = ot()
            if isinstance(val, str):
                return val
        except Exception:
            pass
    if isinstance(ot, str):
        return ot

    # Fallback: walk resp.output items
    output = getattr(resp, "output", None)
    if output and isinstance(output, list):
        parts: List[str] = []
        for item in output:
            try:
                if getattr(item, "type", None) == "message" and getattr(item, "content", None):
                    for c in item.content:
                        if getattr(c, "type", None) == "output_text":
                            t = getattr(c, "text", None)
                            if t:
                                parts.append(str(t))
            except Exception:
                continue
        if parts:
            return "\n".join(parts)

    # last resort
    return str(resp)


def _extract_responses_reasoning_meta(resp: Any) -> Dict[str, Any]:
    """
    Extract minimal reasoning metadata from OpenAI Responses API response.

    This is used to *confirm the model performed internal reasoning* without
    logging any chain-of-thought content.
    """
    meta: Dict[str, Any] = {
        "has_reasoning_output_item": False,
        "reasoning_summary_count": 0,
        "reasoning_encrypted_content_present": False,
        "usage_reasoning_tokens": None,
        "model": getattr(resp, "model", None),
    }
    if resp is None:
        return meta

    # usage.output_tokens_details.reasoning_tokens
    usage = getattr(resp, "usage", None)
    if usage is not None:
        otd = getattr(usage, "output_tokens_details", None)
        if otd is not None:
            rt = getattr(otd, "reasoning_tokens", None)
            meta["usage_reasoning_tokens"] = rt

    output = getattr(resp, "output", None)
    if output and isinstance(output, list):
        for item in output:
            try:
                if getattr(item, "type", None) != "reasoning":
                    continue
                meta["has_reasoning_output_item"] = True
                summary = getattr(item, "summary", None)
                if summary is None:
                    meta["reasoning_summary_count"] = 0
                elif isinstance(summary, list):
                    meta["reasoning_summary_count"] = len(summary)
                elif isinstance(summary, str):
                    meta["reasoning_summary_count"] = 1 if summary.strip() else 0
                else:
                    meta["reasoning_summary_count"] = 1
                enc = getattr(item, "encrypted_content", None)
                meta["reasoning_encrypted_content_present"] = bool(enc)
            except Exception:
                continue

    return meta


def _normalize_keyword(token: Optional[str]) -> Optional[str]:
    """
    将 token 字符串归一化为标准关键词。

    Returns:
        匹配的关键词（小写），或 None 表示无法匹配
    """
    if not token or not isinstance(token, str):
        return None
    tl = token.strip().lower()
    if tl in _LOGP_KEYWORDS:
        return tl
    if "except" in tl:
        return "exceptional"
    return None


def _update_logp_result(logp_result: Dict[str, Optional[float]],
                        token: Optional[str],
                        log_prob: Any) -> None:
    """
    将 (token, log_prob) 写入 logp_result。
    - 若 use_max_per_label=True（默认）：保留同一关键词的最大值（历史行为，会混多位置导致打平）。
    - 若 use_max_per_label=False：仅写入一次，不覆盖（用于只填第一 token 位置时，避免跨位置混用）。
    """
    key = _normalize_keyword(token)
    if key is None or log_prob is None:
        return
    try:
        logp_val = float(log_prob)
    except (TypeError, ValueError):
        return
    existing = logp_result.get(key)
    if existing is None or logp_val > existing:
        logp_result[key] = logp_val


def _fill_logp_from_one_position(logp_result: Dict[str, Optional[float]],
                                  token_probs: Any,
                                  use_max_per_label: bool = False) -> None:
    """
    从单个 token 位置的 top_logprobs 填充 logp_result。
    用于分类任务时只取第一个 token 位置，避免跨位置取 max 造成人为打平。
    use_max_per_label: 同一 label 是否在该位置内取 max（同一位置内一般不会重复 key）。
    """
    if not token_probs:
        return
    top = getattr(token_probs, "top_logprobs", None) if not isinstance(token_probs, dict) else token_probs.get("top_logprobs")
    if not top:
        return
    if isinstance(top, dict):
        for tok, lp in top.items():
            key = _normalize_keyword(tok)
            if key is None or lp is None:
                continue
            try:
                logp_val = float(lp)
            except (TypeError, ValueError):
                continue
            if use_max_per_label:
                existing = logp_result.get(key)
                if existing is None or logp_val > existing:
                    logp_result[key] = logp_val
            else:
                if logp_result.get(key) is None:
                    logp_result[key] = logp_val
    else:
        for item in top:
            if isinstance(item, dict):
                tok, lp = item.get("token"), item.get("logprob")
            else:
                tok, lp = getattr(item, "token", None), getattr(item, "logprob", None)
            key = _normalize_keyword(tok)
            if key is None or lp is None:
                continue
            try:
                logp_val = float(lp)
            except (TypeError, ValueError):
                continue
            if use_max_per_label:
                existing = logp_result.get(key)
                if existing is None or logp_val > existing:
                    logp_result[key] = logp_val
            else:
                if logp_result.get(key) is None:
                    logp_result[key] = logp_val


def _get_provider_type(provider: str) -> str:
    """
    从 provider 路径提取 provider 类型（openai、claude、zai 或 vertex）
    openai / claude / zai 均走 OpenAI 兼容的 chat completions 路径。

    Args:
        provider: Provider路径（如 "openai.openrouter", "claude.official", "zai.official", "vertex.official"）

    Returns:
        Provider类型（"openai", "claude", "zai" 或 "vertex"）
    """
    if not provider:
        return "openai"
    return provider.split(".", 1)[0].lower()


def _get_openai_client(api_key: str, base_url: Optional[str]) -> AsyncOpenAI:
    """获取或创建缓存的 AsyncOpenAI 客户端"""
    cache_key = (api_key, base_url)
    client = _openai_client_cache.get(cache_key)
    if client is None:
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        _openai_client_cache[cache_key] = client
    return client


def _get_vertex_credentials(credentials_path: Optional[str]):
    """获取或缓存 Vertex AI 凭证（线程安全）"""
    if not credentials_path:
        return None
    cached = _credentials_cache.get(credentials_path)
    if cached is not None:
        return cached
    import os
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(f"Service Account 密钥文件不存在: {credentials_path}")
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    _credentials_cache[credentials_path] = credentials
    return credentials


def _ensure_vertex_init(project_id: str, location: str, credentials_path: Optional[str]):
    """确保 Vertex AI 只初始化一次（线程安全，在 executor 中调用）"""
    init_key = (project_id, location, credentials_path)
    if init_key in _vertex_init_done:
        return
    with _vertex_init_lock:
        if init_key in _vertex_init_done:
            return  # double-check after acquiring lock
        credentials = _get_vertex_credentials(credentials_path)
        vertexai.init(project=project_id, location=location, credentials=credentials)
        _vertex_init_done.add(init_key)
        logger.info(f"Vertex AI 初始化完成 (project={project_id}, location={location})")


def _get_vertex_model(model_resource: str, system_instruction: Optional[str]):
    """获取或缓存 GenerativeModel（线程安全，在 executor 中调用）"""
    cache_key = (model_resource, system_instruction)
    cached = _vertex_model_cache.get(cache_key)
    if cached is not None:
        return cached
    with _vertex_model_lock:
        cached = _vertex_model_cache.get(cache_key)
        if cached is not None:
            return cached
        if system_instruction:
            model = GenerativeModel(model_resource, system_instruction=system_instruction)
        else:
            model = GenerativeModel(model_resource)
        _vertex_model_cache[cache_key] = model
        logger.info(f"GenerativeModel 已缓存 (resource={model_resource})")
        return model


async def inference(
    model_config: Dict[str, Any],
    provider_config: Dict[str, Any],
    messages: List[Message],
    enable_logp: bool = False,
    top_logprobs: int = 5,
    **kwargs
) -> Dict[str, Any]:
    """
    统一的推理接口，根据 provider 自动选择 OpenAI 或 Vertex AI 客户端

    Args:
        model_config: 模型配置字典（包含 model_name, model_resource, endpoint 等）
        provider_config: Provider配置字典（包含 provider, api_key, base_url, project_id 等认证信息）
        messages: Message对象列表
        enable_logp: 是否计算logprob
            - OpenAI: 支持，可返回 top_logprobs 个候选 token 的概率
            - Vertex AI: 支持，可返回 logprobs=N 个候选 token 的概率（N 通常为 1-20）
        top_logprobs: 返回的top logprobs数量
            - OpenAI: 直接使用此参数
            - Vertex AI: 映射为 logprobs 参数
        **kwargs: 其他参数（如 temperature, max_tokens 等）

    Returns:
        包含 response_text 和 logp（如果启用）的字典
    """
    provider = provider_config.get("provider")
    if not provider:
        raise ValueError("provider_config 必须包含 provider 字段")

    provider_type = _get_provider_type(provider)

    if provider_type in ("openai", "claude"):
        # claude 使用 Anthropic 的 OpenAI 兼容 API，与 openai 同走 chat completions
        # OpenAI Official uses Responses API for reasoning controls.
        # Keep chat.completions path for logprobs (and for non-official providers).
        if provider == "openai.official" and not enable_logp:
            return await _inference_openai_official_responses(
                model_config=model_config,
                provider_config=provider_config,
                messages=messages,
                **kwargs,
            )
        return await _inference_openai_chat_completions(
            model_config=model_config,
            provider_config=provider_config,
            messages=messages,
            enable_logp=enable_logp,
            top_logprobs=top_logprobs,
            **kwargs,
        )
    elif provider_type == "zai":
        return await _inference_zai_chat_completions(
            model_config=model_config,
            provider_config=provider_config,
            messages=messages,
            enable_logp=enable_logp,
            top_logprobs=top_logprobs,
            **kwargs,
        )
    elif provider_type == "vertex":
        return await _inference_vertex(
            model_config=model_config,
            provider_config=provider_config,
            messages=messages,
            enable_logp=enable_logp,
            top_logprobs=top_logprobs,
            **kwargs
        )
    else:
        raise ValueError(f"不支持的 provider 类型: {provider_type}，请使用 openai.*、claude.*、zai.* 或 vertex")


async def _inference_openai_official_responses(
    model_config: Dict[str, Any],
    provider_config: Dict[str, Any],
    messages: List[Message],
    **kwargs,
) -> Dict[str, Any]:
    """
    OpenAI Official inference via Responses API.

    This path exists because GPT-5.x reasoning controls are exposed via
    `reasoning.effort` / `text.verbosity` on the Responses API.

    Notes:
    - We intentionally do NOT support logprobs here (callers should use enable_logp=True
      which routes to chat.completions).
    - By default, enable_thinking=True maps to reasoning.effort="high".
    """
    model_name = model_config.get("model_name")
    if not model_name:
        raise ValueError("model_config 中必须包含 model_name 字段（OpenAI 使用）")

    api_key = provider_config.get("api_key")
    base_url = provider_config.get("base_url")

    if not api_key or api_key == "EMPTY" or api_key.strip() == "":
        api_key = "sk-local"

    client = _get_openai_client(api_key, base_url)
    messages_dict = [msg.to_dict() for msg in messages]

    enable_thinking = bool(kwargs.get("enable_thinking", False))
    effort = kwargs.get("reasoning_effort")
    if effort is None and enable_thinking:
        effort = "high"
    verbosity = kwargs.get("text_verbosity")

    # Only attach reasoning/text configs when the caller explicitly needs them;
    # regular (non-reasoning) models reject these fields entirely.
    reasoning_cfg = {"effort": effort} if effort is not None else None
    text_cfg = {"verbosity": verbosity} if verbosity else None

    # Responses API 也支持温度（若未提供则使用默认）。
    temperature = kwargs.get("temperature", None)

    extra_kw: Dict[str, Any] = {}
    if reasoning_cfg is not None:
        extra_kw["reasoning"] = reasoning_cfg
    if text_cfg is not None:
        extra_kw["text"] = text_cfg
    if temperature is not None:
        extra_kw["temperature"] = temperature

    # Prefer structured inputs; fall back to concatenated string on SDK/provider mismatch.
    try:
        resp = await client.responses.create(
            model=model_name,
            input=messages_dict,
            **extra_kw,
        )
    except Exception:
        chunks: List[str] = []
        for m in messages_dict:
            role = (m.get("role") or "").upper()
            content = m.get("content") or ""
            if content:
                chunks.append(f"[{role}]\n{content}")
        joined = "\n\n".join(chunks)
        resp = await client.responses.create(
            model=model_name,
            input=joined,
            **extra_kw,
        )

    response_text = _extract_responses_output_text(resp)
    reasoning_meta = _extract_responses_reasoning_meta(resp)
    # 完整 response 对象（整份 API 返回），供 validation_results.jsonl 保存
    full_response = _openai_response_to_dict(resp)
    logger.info(
        "OpenAI Responses reasoning_meta: "
        f"has_reasoning_output_item={reasoning_meta.get('has_reasoning_output_item')}, "
        f"reasoning_summary_count={reasoning_meta.get('reasoning_summary_count')}, "
        f"usage_reasoning_tokens={reasoning_meta.get('usage_reasoning_tokens')}"
    )
    logger.info(f"OpenAI Responses output: {response_text}")
    return {
        "response_text": response_text,
        "reasoning_meta": reasoning_meta,
        "full_response": full_response,
    }


async def _inference_openai_chat_completions(
    model_config: Dict[str, Any],
    provider_config: Dict[str, Any],
    messages: List[Message],
    enable_logp: bool = False,
    top_logprobs: int = 5,
    **kwargs
) -> Dict[str, Any]:
    """
    使用 OpenAI 客户端进行推理

    Args:
        model_config: 模型配置字典（包含 model_name, model_resource, endpoint 等）
        provider_config: Provider配置字典（包含 api_key, base_url 等认证信息）
        messages: Message对象列表
        enable_logp: 是否计算logprob
        top_logprobs: 返回的top logprobs数量
        **kwargs: 其他参数

    Returns:
        包含 response_text 和 logp（如果启用）的字典
    """
    model_name = model_config.get("model_name")
    if not model_name:
        raise ValueError("model_config 中必须包含 model_name 字段（OpenAI 使用）")

    api_key = provider_config.get("api_key")
    base_url = provider_config.get("base_url")

    # 处理本地服务器的 api_key（如果为 "EMPTY" 或 None，使用占位符）
    if not api_key or api_key == "EMPTY" or api_key.strip() == "":
        api_key = "sk-local"

    client = _get_openai_client(api_key, base_url)

    messages_dict = [msg.to_dict() for msg in messages]

    # 过滤掉 Vertex AI 专用的 kwargs，避免传递给 OpenAI API 导致报错
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in _VERTEX_ONLY_KWARGS and k not in _OPENAI_INTERNAL_KWARGS}

    params = {
        "model": model_name,
        "messages": messages_dict,
        **filtered_kwargs
    }

    if enable_logp:
        params["logprobs"] = True
        # OpenRouter: top_logprobs must be between 0 and 20 (see openrouter.ai/docs/api-reference/parameters)
        _top = top_logprobs
        if base_url and "openrouter" in (base_url or "").lower():
            _top = min(20, max(0, _top))
        params["top_logprobs"] = _top

    response = await client.chat.completions.create(**params)
    choice = response.choices[0]
    response_text = choice.message.content
    logger.info(f"OpenAI output: {response_text}")

    # 完整 response 对象（整份 API 返回），供 validation_results.jsonl 保存
    result: Dict[str, Any] = {
        "response_text": response_text,
        "full_response": _openai_response_to_dict(response),
    }

    # 如果启用 logp，始终设置 logp 键（保证与 Vertex 路径一致）
    if enable_logp:
        logp_result = {k: None for k in _LOGP_KEYWORDS}

        if not (getattr(choice, "logprobs", None) and getattr(choice.logprobs, "content", None)):
            if base_url and "openrouter" in (base_url or "").lower():
                logger.warning(
                    "OpenRouter: logprobs requested but response has no choice.logprobs.content; "
                    "the model may not support logprobs on OpenRouter."
                )

        # 分类任务只取第一个 token 位置的 top_logprobs，避免跨位置取 max 导致人为打平
        if choice.logprobs and choice.logprobs.content:
            first = choice.logprobs.content[0]
            _fill_logp_from_one_position(logp_result, first, use_max_per_label=False)

        result["logp"] = logp_result

    return result


async def _inference_zai_chat_completions(
    model_config: Dict[str, Any],
    provider_config: Dict[str, Any],
    messages: List[Message],
    enable_logp: bool = False,
    top_logprobs: int = 5,
    **kwargs,
) -> Dict[str, Any]:
    """
    ZAI (智谱) OpenAI-compat Chat Completions 推理。

    Endpoint（示例）:
      POST https://open.bigmodel.cn/api/paas/v4/chat/completions

    说明：
    - thinking 控制使用 ZAI 的 `thinking` 字段（不是 OpenRouter 的 reasoning）。
    - logprobs 是否支持取决于 ZAI 侧实现；此处做 best-effort 解析（不存在则返回 None）。
    """
    import httpx

    model_name = model_config.get("model_name")
    if not model_name:
        raise ValueError("model_config 中必须包含 model_name 字段（ZAI 使用）")

    api_key = (provider_config.get("api_key") or "").strip()
    base_url = (provider_config.get("base_url") or "").strip().rstrip("/")
    if not base_url:
        base_url = "https://open.bigmodel.cn/api/paas/v4"

    if not api_key or api_key == "EMPTY":
        raise ValueError("ZAI api_key 未配置（providers.zai.official.api_key 或环境变量）")

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages_dict = [msg.to_dict() for msg in messages]

    # 从 kwargs 里提取 extra_body（用于注入 thinking 等 provider 特有字段）
    extra_body = kwargs.get("extra_body")
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in _VERTEX_ONLY_KWARGS and k not in _OPENAI_INTERNAL_KWARGS and k != "extra_body"}

    body: Dict[str, Any] = {
        "model": model_name,
        "messages": messages_dict,
        **filtered_kwargs,
    }

    # 如果上层通过 extra_body 注入 provider 特有字段（如 thinking），合并到 body（extra_body 优先）
    if isinstance(extra_body, dict):
        body.update(extra_body)

    # 兜底：若上层仅传 enable_thinking（项目内部），且没有显式指定 thinking，则按 ZAI 约定补一个开关
    # （避免把 OpenAI 的 reasoning 误传给 ZAI）
    if "thinking" not in body and "enable_thinking" in kwargs:
        if bool(kwargs.get("enable_thinking")):
            body["thinking"] = {"type": "enabled"}
        else:
            body["thinking"] = {"type": "disabled"}

    timeout = float(kwargs.get("timeout", 120.0))
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=body)
        # 兼容错误信息
        if resp.status_code >= 400:
            raise RuntimeError(f"ZAI HTTP {resp.status_code}: {resp.text}")
        data = resp.json()

    # 解析 content
    choices = data.get("choices") or []
    msg0 = (choices[0].get("message") or {}) if choices else {}
    response_text = msg0.get("content")
    if response_text is None:
        # 少数实现用 choices[0].text
        response_text = choices[0].get("text") if choices else ""

    result: Dict[str, Any] = {
        "response_text": response_text or "",
        "full_response": data,
    }

    # best-effort logp 解析（若 ZAI 不返回则为 None）
    if enable_logp:
        logp_result = {k: None for k in _LOGP_KEYWORDS}
        try:
            lp = (choices[0] or {}).get("logprobs") if choices else None
            content_lps = (lp or {}).get("content") if isinstance(lp, dict) else None
            if content_lps:
                for token_probs in content_lps:
                    top = token_probs.get("top_logprobs") if isinstance(token_probs, dict) else None
                    if not top:
                        continue
                    if isinstance(top, dict):
                        for tok, lpv in top.items():
                            _update_logp_result(logp_result, tok, lpv)
                    else:
                        for item in top:
                            if isinstance(item, dict):
                                _update_logp_result(logp_result, item.get("token"), item.get("logprob"))
        except Exception:
            # 解析失败不影响主流程
            pass
        result["logp"] = logp_result

    return result


async def _inference_vertex(
    model_config: Dict[str, Any],
    provider_config: Dict[str, Any],
    messages: List[Message],
    enable_logp: bool = False,
    top_logprobs: int = 5,
    **kwargs
) -> Dict[str, Any]:
    """
    使用 Vertex AI 客户端进行推理

    Args:
        model_config: 模型配置字典（包含 model_name, model_resource, endpoint 等）
        provider_config: Provider配置字典（包含 project_id, location, credentials_path 等认证信息）
        messages: Message对象列表
        enable_logp: 是否计算logprob（Vertex AI 支持完整的 logprobs，包括 top_logprobs）
        top_logprobs: 返回的top logprobs数量（映射为 Vertex AI 的 logprobs 参数，通常为 1-20）
        **kwargs: 其他参数

    Returns:
        包含 response_text 和 logp（如果启用）的字典
    """
    model_resource = model_config.get("endpoint") or model_config.get("model_resource")
    if not model_resource:
        raise ValueError("model_config 中必须包含 endpoint（微调模型）或 model_resource（基础模型）字段（Vertex AI 使用）")

    if not VERTEX_AVAILABLE:
        raise ImportError("vertexai 未安装，请安装: pip install google-cloud-aiplatform vertexai")

    import os
    import asyncio

    # 从 provider_config 获取配置
    project_id = provider_config.get("project_id") or os.getenv("GOOGLE_CLOUD_PROJECT")
    location = provider_config.get("location", "us-central1")
    credentials_path = provider_config.get("credentials_path") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not project_id:
        raise ValueError("Vertex AI project_id 未配置，请在配置文件中设置或设置 GOOGLE_CLOUD_PROJECT 环境变量")

    loop = asyncio.get_running_loop()

    # 初始化 Vertex AI（仅首次调用时执行，后续直接跳过）
    await loop.run_in_executor(None, _ensure_vertex_init, project_id, location, credentials_path)

    # 将 messages 转换为 Vertex AI 格式
    system_instruction = None
    user_parts = []

    for msg in messages:
        if msg.role == MessageRole.SYSTEM:
            if system_instruction is None:
                system_instruction = msg.content
            else:
                system_instruction = f"{system_instruction}\n\n{msg.content}"
        elif msg.role == MessageRole.USER:
            user_parts.append(msg.content)
        elif msg.role in (MessageRole.ASSISTANT, MessageRole.MODEL):
            user_parts.append(msg.content)

    if not user_parts:
        raise ValueError("messages 中必须包含至少一个 USER 或 ASSISTANT 消息")

    user_content = "\n\n".join(user_parts) if len(user_parts) > 1 else user_parts[0]

    if not user_content or not user_content.strip():
        raise ValueError("user content 不能为空")

    # 准备生成配置
    generation_config_dict = kwargs.get("generation_config", {})

    if enable_logp:
        logprobs_count = max(1, min(top_logprobs, 20))
        config_params = {
            "response_logprobs": True,
            "logprobs": logprobs_count,
        }
        for key, value in generation_config_dict.items():
            if key not in ("response_logprobs", "logprobs", "temperature"):
                config_params[key] = value
        generation_config = GenerationConfig(**config_params)
    elif generation_config_dict:
        generation_config = GenerationConfig(**generation_config_dict)
    else:
        generation_config = None

    # 记录输入信息
    logger.info("=" * 80)
    logger.info("Vertex AI Inference Input:")
    logger.info(f"  model_resource: {model_resource}")
    if system_instruction:
        si_preview = system_instruction[:100] + "..." if len(system_instruction) > 100 else system_instruction
        logger.info(f"  system_instruction: {si_preview}")
        logger.info(f"  system_instruction length: {len(system_instruction)}")
    uc_preview = user_content[:100] + "..." if len(user_content) > 100 else user_content
    logger.info(f"  user_content: {uc_preview}")
    logger.info(f"  user_content length: {len(user_content)}")
    if generation_config:
        logger.info(f"  generation_config: {generation_config}")
    logger.info(f"  enable_logp: {enable_logp}")
    if enable_logp:
        logger.info(f"  top_logprobs: {top_logprobs}")
    logger.info("=" * 80)

    # 获取缓存的 GenerativeModel 并生成内容（同步调用，在线程池中运行）
    def _generate():
        try:
            model = _get_vertex_model(model_resource, system_instruction)

            if generation_config:
                response = model.generate_content(user_content, generation_config=generation_config)
            else:
                response = model.generate_content(user_content)

            return response
        except Exception as e:
            logger.error(f"Vertex AI generate_content 失败: {e}")
            logger.error(f"  model_resource: {model_resource}")
            logger.error(f"  system_instruction length: {len(system_instruction) if system_instruction else 0}")
            logger.error(f"  user_content length: {len(user_content) if user_content else 0}")
            logger.error(f"  generation_config: {generation_config}")
            raise

    timeout = kwargs.get("timeout", 120.0)
    try:
        response = await asyncio.wait_for(loop.run_in_executor(None, _generate), timeout=timeout)
    except asyncio.TimeoutError:
        logger.error(f"Vertex AI generate_content 超时（{timeout}秒）")
        raise TimeoutError(f"Vertex AI 调用超时（{timeout}秒）")

    # 提取响应文本和 reasoning content
    response_text = response.text if hasattr(response, 'text') else str(response)
    reasoning_content = None

    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
            parts = candidate.content.parts
            reasoning_parts = []
            text_parts = []

            for part in parts:
                if hasattr(part, 'thought') and part.thought:
                    # Vertex AI 明确标记的 thinking/reasoning part
                    text = getattr(part, 'text', '')
                    if text:
                        reasoning_parts.append(text)
                elif hasattr(part, 'text'):
                    text_parts.append(getattr(part, 'text', ''))

            if reasoning_parts:
                reasoning_content = "\n".join(reasoning_parts)
                # 如果有 reasoning parts，用 text_parts 重建 response_text
                if text_parts:
                    response_text = text_parts[-1]

    # 记录输出信息
    logger.info("=" * 80)
    logger.info("Vertex AI Inference Output:")
    logger.info(f"  response_text: {response_text}")
    logger.info(f"  response_text length: {len(response_text) if response_text else 0}")
    if reasoning_content:
        rc_preview = reasoning_content[:200] + "..." if len(reasoning_content) > 200 else reasoning_content
        logger.info(f"  reasoning_content: {rc_preview}")
        logger.info(f"  reasoning_content length: {len(reasoning_content)}")

    if enable_logp and hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'logprobs_result') and candidate.logprobs_result:
            logprobs_data = candidate.logprobs_result
            if hasattr(logprobs_data, 'chosen_candidates'):
                chosen = logprobs_data.chosen_candidates
                logger.info(f"  logprobs_result.chosen_candidates count: {len(chosen) if chosen else 0}")
                if chosen:
                    logger.info("  First 5 tokens:")
                    for i, token_info in enumerate(list(chosen)[:5]):
                        token_text = getattr(token_info, 'token', None)
                        logp = getattr(token_info, 'log_probability', None)
                        token_id = getattr(token_info, 'token_id', None)
                        logger.info(f"    [{i+1}] token={token_text!r} log_prob={logp} token_id={token_id}")
            if hasattr(logprobs_data, 'top_candidates'):
                top = logprobs_data.top_candidates
                logger.info(f"  logprobs_result.top_candidates count: {len(top) if top else 0}")
    logger.info("=" * 80)

    result: Dict[str, Any] = {
        "response_text": response_text
    }

    if reasoning_content:
        result["reasoning_content"] = reasoning_content

    # 如果启用 logp，始终设置 logp 键
    if enable_logp:
        logp_result = {k: None for k in _LOGP_KEYWORDS}

        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'logprobs_result') and candidate.logprobs_result:
                logprobs_data = candidate.logprobs_result

                # 从 chosen_candidates 提取
                if hasattr(logprobs_data, 'chosen_candidates') and logprobs_data.chosen_candidates:
                    for item in logprobs_data.chosen_candidates:
                        token = getattr(item, 'token', None)
                        log_prob = getattr(item, 'log_probability', None)
                        # 处理 token 可能是 bytes
                        if isinstance(token, bytes):
                            token = token.decode('utf-8', errors='replace')
                        elif token is not None and not isinstance(token, str):
                            token = str(token)
                        _update_logp_result(logp_result, token, log_prob)

                # 从 top_candidates 提取（当 top_logprobs > 1 时）
                if top_logprobs > 1 and hasattr(logprobs_data, 'top_candidates') and logprobs_data.top_candidates:
                    for top_item in logprobs_data.top_candidates:
                        if not hasattr(top_item, 'candidates'):
                            continue
                        for candidate_token in top_item.candidates:
                            token = getattr(candidate_token, 'token', None)
                            log_prob = getattr(candidate_token, 'log_probability', None)
                            if isinstance(token, bytes):
                                token = token.decode('utf-8', errors='replace')
                            elif token is not None and not isinstance(token, str):
                                token = str(token)
                            _update_logp_result(logp_result, token, log_prob)

        result["logp"] = logp_result
        if any(v is not None for v in logp_result.values()):
            logger.info(f"Vertex AI logp 返回结果: {logp_result}")

    return result
