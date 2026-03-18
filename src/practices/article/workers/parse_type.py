"""
解析文章类型的 worker
"""
import asyncio
import logging
import os
import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中（用于直接运行或模块运行）
_file_path = Path(__file__).resolve()
# parse_type.py 在 src/practices/article/workers/，需要向上5级到项目根目录
_project_root = _file_path.parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import json5 as json
from openai import AsyncOpenAI

# 处理相对导入（支持直接运行和模块运行）
try:
    from ..models import Article, init_database
    from ..utils import task_generator
except ImportError:
    # 直接运行时，使用绝对导入
    from practices.article.models import Article, init_database
    from practices.article.utils import task_generator

# 配置日志
logger = logging.getLogger(__name__)

    # 导入配置函数（使用 core 中的新配置逻辑）
try:
    from src.core.utils.config import ConfigLoader
except ImportError:
    # 如果导入失败，使用环境变量
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


def create_task_exception_handler(article: Article):
    """创建一个能访问 Article 的异常处理器"""
    def handle_task_exception(asyncio_task):
        try:
            exc = asyncio_task.exception()
            if exc:
                logger.error(f"Background task error: {exc}")      
                asyncio.create_task(article.save())
        except Exception as e:
            logger.error(f"Error in exception handler: {e}")    
    return handle_task_exception

EXTRACTION_PROMPT = """
    You are a classification assistant. Your task is to determine the **type of academic paper** based on its abstract.

    ### Output format (JSON only)
    Return **only** a valid JSON object (no extra text or explanations):

    ```json
    {
        "type": "Enum(review | study | others)"
    }
    ```

    ### Classification Rules

    You must classify the paper into **exactly one** of the following three categories:

    1. **review** - systematic reviews, literature reviews, meta-analyses, theoretical papers that synthesize prior work.
    2. **study** - original empirical research (qualitative, quantitative, experimental, or computational studies).
    3. **others** - all other types not fitting review or study.

    If the paper's type cannot be confidently determined, use `"type": "others"`.
    """


async def _parse_type(article: Article):
    """解析文章类型：优先用 abstract，无则退化到 title。"""
    abstract = article.entries.get("abstract", "") if article.entries else ""
    title = (article.title or "").strip()
    if abstract:
        text_for_classification = abstract.strip()
        text_label = "Abstract"
    elif title:
        text_for_classification = title
        text_label = "Title"
        logger.info(f"Article {article.id} has no abstract, using title for type classification")
    else:
        logger.warning(f"Article {article.id} has no abstract nor title, setting type to failed")
        article.type = "failed"
        await article.save()
        return

    model_name = "qwen/qwen3-30b-a3b-instruct-2507"
    config_path = "assets/model.toml"
    provider = "openai.openrouter"  # 根据实际情况设置 provider
    
    # 获取Provider配置
    merged_config = ConfigLoader.get_provider_config_with_env_fallback(config_path, provider)
    api_key = merged_config.get("api_key") if merged_config else None
    base_url = merged_config.get("base_url") if merged_config else None
    
    if not api_key:
        logger.error(f"未找到模型 {model_name} 的 API key")
        article.type = "failed"
        await article.save()
        return
    
    # 处理本地服务器的 api_key（如果为 "EMPTY" 或 None，使用占位符）
    if not api_key or api_key == "EMPTY" or api_key.strip() == "":
        api_key = "sk-local"
    
    # 创建 OpenAI 客户端
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    
    # 准备请求参数
    params = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": f"{text_label}: {text_for_classification}\n\n### Please extract the type in JSON format."},
        ],
        "response_format": {"type": "json_object"}
    }
    
    try:
        # 调用 OpenAI API
        response = await client.chat.completions.create(**params)
        response_text = response.choices[0].message.content
        
        # 解析 JSON 响应
        response_data = json.loads(response_text)
        type_value = response_data.get("type", "").strip() if response_data.get("type") else None

        # 清理 null 字符串
        if type_value in ["null", "None", "N/A", ""]:
            type_value = None
        
        article.type = type_value
        article.trial_count = 0
        await article.save()
        
        logger.info(f"type parsing completed: type={type_value}")
        
    except Exception as e:
        logger.error(f"type processing failed: {e}")
        article.type = "failed"
        await article.save()


async def article_parse_type_worker(connection_string: str):
    """解析文章类型的worker"""
    logger.info(f"Article parse type worker started")
    await init_database(connection_string=connection_string)
    
    # 检查是否有符合条件的文档
    # 处理 type='space' 且 status='created' 的新文档
    # 处理 type='failed' 的失败文档（无论 status 是什么，都需要重试）
    filter_condition = {
        "$or": [
            {"type": "space", "status": "created"},
            {"type": "failed", "status": {"$ne": "parsing_type"}},
        ]
    }
    count = await Article.find(filter_condition).count()
    logger.info(f"找到 {count} 个符合条件的文档 (type='space'&status='created' 或 type='failed')")
    
    if count == 0:
        logger.warning("没有符合条件的文档，worker 将等待新文档...")
    
    async for article in task_generator(
        Article,
        filter=filter_condition,
        set={"status": "parsing_type"},  # 原子性设置状态为正在解析类型
        inc={},  # 可以根据需要递增字段（如 trial_count）
        sleep_time=0.1
    ):
        logger.info(f"开始处理文章: {article.id} (title: {article.title[:50] if article.title else 'N/A'}...)")
        bg_task = asyncio.create_task(_parse_type(article))
        bg_task.add_done_callback(create_task_exception_handler(article))


if __name__ == "__main__":
    """直接运行 worker"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 读取数据库配置
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
    db_name = default_config.get('db_name', 'RQ')
    conn_str = default_config.get('connection_string', '')
    
    # 替换<DBNAME>占位符
    if '<DBNAME>' in conn_str:
        connection_string = conn_str.replace('<DBNAME>', db_name)
    else:
        connection_string = conn_str
    
    if not connection_string:
        logger.error("无法获取数据库连接字符串")
        sys.exit(1)
    
    logger.info(f"使用数据库: {db_name}")
    logger.info(f"连接字符串: {connection_string}")
    
    # 运行 worker
    try:
        asyncio.run(article_parse_type_worker(connection_string))
    except KeyboardInterrupt:
        logger.info("Worker 已停止")

