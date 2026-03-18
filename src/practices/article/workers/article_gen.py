"""
生成文章 entries 的 worker
"""
import asyncio
import logging
import os
import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中（用于直接运行或模块运行）
_file_path = Path(__file__).resolve()
# article_gen.py 在 src/practices/article/workers/，需要向上5级到项目根目录
_project_root = _file_path.parent.parent.parent.parent.parent
_src_root = _project_root / "src"
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

import json5 as json

# 处理相对导入（支持直接运行和模块运行）
try:
    from ..models import Article, init_database
    from ..ob import ARTICLE_EXTRACTION_PROMPT
    from ..utils import task_generator, merge_context
    from src.core.schema.message import Message, MessageRole
    from src.core.utils.inference import inference
except ImportError:
    from practices.article.models import Article, init_database
    from practices.article.ob import ARTICLE_EXTRACTION_PROMPT
    from practices.article.utils import task_generator, merge_context
    from core.schema.message import Message, MessageRole
    from core.utils.inference import inference

# 配置日志
logger = logging.getLogger(__name__)

# 导入配置函数（使用 core 中的新配置逻辑）
try:
    from src.core.utils.config import ConfigLoader
except ImportError:
    class ConfigLoader:
        @staticmethod
        def get_provider_config_with_env_fallback(config_path: str, provider: str):
            """Fallback: 返回环境变量配置"""
            logger.warning(f"无法导入配置模块，使用环境变量")
            return {
                "api_key": os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("VERTEX_API_KEY"),
                "base_url": os.getenv("OPENAI_BASE_URL") or os.getenv("GOOGLE_BASE_URL") or os.getenv("VERTEX_BASE_URL"),
                "provider": provider
            }


async def _retrieve_context(article: Article) -> str:
    """获取PDF内容，用于生成 entries"""
    content = article.content or ""
    return merge_context(f"### PDF text content:\n{content}")


async def _generate_entries(article: Article):
    """生成文章的5个版本摘要"""
    context = await _retrieve_context(article)

    model_name = os.getenv("ARTICLE_GEN_MODEL", "qwen/qwen3-235b-a22b-2507")
    config_path = "assets/model.toml"
    provider = os.getenv("ARTICLE_GEN_PROVIDER", "openai.openrouter")

    merged_config = ConfigLoader.get_provider_config_with_env_fallback(config_path, provider)
    api_key = merged_config.get("api_key") if merged_config else None
    base_url = merged_config.get("base_url") if merged_config else None
    trial_times = int(os.getenv("TRIAL_TIMES", 1))

    if not context.strip():
        logger.error("article content 为空，无法生成 entries: %s", article.id)
        article.status = "entries_failed"
        await article.save()
        return

    if not api_key:
        logger.error(f"未找到模型 {model_name} 的 API key")
        article.status = "entries_failed"
        await article.save()
        return

    if api_key == "EMPTY" or api_key.strip() == "":
        api_key = "sk-local"

    try:
        result = await inference(
            model_config={"model_name": model_name},
            provider_config=merged_config,
            messages=[
                Message(role=MessageRole.SYSTEM, content=ARTICLE_EXTRACTION_PROMPT),
                Message(
                    role=MessageRole.USER,
                    content=context + "\n\n### Please generate all 5 versions in JSON format.",
                ),
            ],
            response_format={"type": "json_object"},
        )
        response_text = result.get("response_text") or ""
        response_data = json.loads(response_text)

        required_fields = [
            "core_rq_short",
            "rq_with_context",
            "gap_focused",
            "theory_and_model",
            "contribution_focused",
        ]

        entries = {}
        for field in required_fields:
            value = response_data.get(field, "").strip() if response_data.get(field) else None
            if not value or value in ["null", "None", "N/A", ""]:
                logger.warning(f"Missing or invalid field: {field}")
                entries[field] = None
            else:
                entries[field] = value

        article.entries.update(entries)
        article.status = "entries_parsed"
        article.trial_count = 0
        await article.save()

        logger.info(f"entries generation completed for article: {article.title}")

    except Exception as e:
        logger.error(f"entries generation processing failed: {e}")
        article.status = "entries_retrying" if article.trial_count < trial_times else "entries_failed"
        await article.save()


async def article_generate_entries_worker(connection_string: str):
    """文章实体生成 worker"""
    logger.info(f"Article generate entries worker started")
    await init_database(connection_string=connection_string)

    trial_times = int(os.getenv("TRIAL_TIMES", 1))
    worker_concurrency = max(1, int(os.getenv("WORKER_CONCURRENCY", "50")))
    launch_interval = float(os.getenv("ENTRY_LAUNCH_INTERVAL", "0.02"))
    filter_condition = {
        "$or": [
            {"status": "content_parsed", "type": "study"},
            {"status": "entries_retrying", "type": "study", "trial_count": {"$lt": trial_times}},
        ]
    }
    count = await Article.find(filter_condition).count()
    logger.info("entries worker concurrency=%d", worker_concurrency)
    logger.info("entries launch interval=%ss", launch_interval)
    logger.info(f"找到 {count} 个符合条件的文档 (status='content_parsed' 且 type='study' 或 'entries_retrying')")

    if count == 0:
        logger.warning("没有符合条件的文档，worker 将等待新文档...")

    in_flight: set[asyncio.Task] = set()

    async def _run_one(target_article: Article):
        await _generate_entries(target_article)

    async for article in task_generator(
        Article,
        filter=filter_condition,
        set={"status": "entries_parsing"},
        inc={"trial_count": 1},
        sleep_time=int(os.getenv("SLEEP_TIME", 30)),
    ):
        logger.info(f"开始处理文章: {article.id} (title: {article.title[:50] if article.title else 'N/A'}...)")
        task = asyncio.create_task(_run_one(article))
        in_flight.add(task)
        task.add_done_callback(in_flight.discard)

        if launch_interval > 0:
            await asyncio.sleep(launch_interval)

        if len(in_flight) >= worker_concurrency:
            await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)


def _run_from_cli() -> None:
    """直接运行 worker"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        import tomllib  # Python 3.11+
    except ImportError:
        import tomli as tomllib  # fallback for older Python versions

    config_path = _project_root / "assets" / "database.toml"
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        sys.exit(1)

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    default_config = config.get('default', {})
    db_name = os.getenv("ARTICLE_DB_NAME") or default_config.get('db_name', 'RQ')
    conn_str = default_config.get('connection_string', '')

    if '<DBNAME>' in conn_str:
        connection_string = conn_str.replace('<DBNAME>', db_name)
    else:
        connection_string = conn_str

    if not connection_string:
        logger.error("无法获取数据库连接字符串")
        sys.exit(1)

    logger.info(f"使用数据库: {db_name}")
    logger.info(f"连接字符串: {connection_string}")

    try:
        asyncio.run(article_generate_entries_worker(connection_string))
    except KeyboardInterrupt:
        logger.info("Worker 已停止")


PROMPT = """
    # ROLE
    You are an objective research paper analyzer. Your task is to extract and present research questions and core elements from academic papers WITHOUT interpretation, embellishment, or improvement.

    # CRITICAL PRINCIPLE: OBJECTIVITY OVER PERSUASIVENESS
    - Present the paper EXACTLY as written by the authors
    - Do NOT add theoretical sophistication if it's not there
    - Do NOT create compelling hooks if the original lacks them
    - Do NOT infer contributions beyond what authors explicitly state
    - Do NOT improve weak framing - describe it as presented
    - If the idea seems underdeveloped in the original, your summary should reflect that

    Your goal: Represent the research proposal exactly as the authors present it—the way a doctoral student would pitch their idea to an advisor. Convey their thinking faithfully, including any lack of polish or theoretical sophistication, so the professor can understand and evaluate the original idea.

    ---

    # OUTPUT STRUCTURE
    Generate exactly 5 versions in JSON format:

    ---

    ## VERSION 1: CORE_RQ_SHORT
    **Purpose:** Distill the essential research question(s)
    **Word count:** 40-60 words (2-3 sentences maximum)
    **Structure:**
    - Sentence 1: The phenomenon or behavior under study
    - Sentence 2: The specific question or what's being tested
    - [Optional Sentence 3: The key boundary condition or mechanism if central to RQ]

    ---

    ## VERSION 2: RQ_WITH_CONTEXT
    **Purpose:** Add just enough context for a professor to evaluate the idea's merit
    **Word count:** 120-150 words (1 paragraph)
    **Structure:**
    - What phenomenon/problem (1-2 sentences)
    - What's missing/unclear in existing research - the gap (2-3 sentences)  
    - The research question (1-2 sentences)
    - The approach/framework used (1 sentence)
    - Key claimed contribution (1 sentence)

    ---

    ## VERSION 3: GAP_FOCUSED
    **Purpose:** Emphasize what's unknown and how this study addresses it
    **Word count:** 100-130 words (1 paragraph)
    **Structure:**
    - What existing research has established (2 sentences)
    - What remains unknown/unresolved (2-3 sentences)
    - How this study addresses the gap/extends the prior research/challenges the understanding (2 sentences)
    - Expected insight (1 sentence)

    ---

    ## VERSION 4: THEORY_AND_MODEL
    **Purpose:** Describe the theoretical framework and research model
    **Word count:** 100-130 words (1 paragraph)
    **Structure:**
    - Core theoretical lens/framework (1-2 sentences)
    - How theory is applied to the phenomenon (2 sentences)
    - Key variables and relationships (2-3 sentences)
    - Theoretical contribution claimed (1 sentence)

    ---

    ## VERSION 5: CONTRIBUTION_FOCUSED  
    **Purpose:** Extract what the authors claim as their contributions
    **Word count:** 80-100 words
    **Structure:**
    - Primary theoretical contribution (1-2 sentences)
    - Empirical/methodological contribution if claimed (1 sentence)
    - Practical contribution if claimed (1 sentence)
    - How it advances the literature (1-2 sentences)

    ---

    # EXTRACTION RULES

    ## Where to Look:
    Focus on the **front-end** of the paper:
    - **Abstract**
    - **Introduction** (entire section - contains RQ, gap, motivation)
    - **Theoretical Development** (theory and hypotheses framing)

    Most information needed is in these sections. Do NOT need to read results/discussion unless contribution statements are unclear.

    ## What to Extract:
    1. **Research Questions:** Usually in abstract's and introduction
    2. **Gaps/problematization:** mostly in introduction and sometimes in theoretical development
    3. **Theory:** introduced in introduction and often elaborated in theory development sections
    4. **Contributions:** Abstract, introduction's end

    ## What to Avoid:
    ❌ Adding your own theoretical connections
    ❌ Improving vague or weak language
    ❌ Creating persuasive hooks not in the original
    ❌ Inferring contributions not explicitly stated
    ❌ Making gaps sound more compelling than presented

    ## Language Rules:
    ✅ Use the authors' exact terminology for key constructs
    ✅ Preserve the level of theoretical sophistication in the original
    ✅ Match the certainty level (e.g., "explores" vs. "demonstrates")
    ✅ If authors use simple language, you use simple language

    ---

    # JSON OUTPUT FORMAT

    Output the following JSON structure with all 5 versions:
    ```json
    {
    "core_rq_short": "string",
    "rq_with_context": "string",
    "gap_focused": "string",
    "theory_and_model": "string",
    "contribution_focused": "string"
    }
    ```

    """


if __name__ == "__main__":
    _run_from_cli()
