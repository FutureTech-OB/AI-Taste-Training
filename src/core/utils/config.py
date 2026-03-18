"""
配置加载工具 - Provider 和 Model 分离架构
配置结构：
[providers.openai.openrouter]
api_key = "..."
base_url = "..."

[providers.openai.official]
api_key = "..."
base_url = "..."

[providers.vertex.official]
project_id = "..."
location = "..."
credentials_path = "..."

[models.gemini-2.5pro-OB]  # Model 配置（独立于 providers）
provider = "vertex.official"  # 引用 provider
model_resource = "projects/695868881031/locations/us-central1/models/5972330558288560128@1"

注意：
- Provider 配置：两级路径（如 "openai.openrouter", "vertex.official"）
- Model 配置：单级路径（如 "gemini-2.5pro-OB"），包含 provider 引用和 model_resource
"""
import os
import logging
from typing import Dict, Any, Optional

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # fallback for older Python versions

try:
    import toml
except ImportError:
    toml = None

logger = logging.getLogger(__name__)

# 全局配置缓存：{config_path: config_dict}
_config_cache: Dict[str, Dict[str, Any]] = {}


class ConfigLoader:
    """配置加载器（Provider 架构，支持两级Provider分类）"""

    @staticmethod
    def load_toml(file_path: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        加载TOML配置文件（带缓存）

        Args:
            file_path: 配置文件路径
            use_cache: 是否使用缓存

        Returns:
            配置字典
        """
        # 使用缓存
        if use_cache and file_path in _config_cache:
            return _config_cache[file_path]
        
        if not os.path.exists(file_path):
            logger.warning(f"配置文件 {file_path} 不存在")
            return {}
        
        try:
            if toml is not None:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = toml.load(f)
            else:
                with open(file_path, "rb") as f:
                    config = tomllib.load(f)
            
            # 缓存配置
            if use_cache:
                _config_cache[file_path] = config
                logger.info(f"成功加载配置文件: {file_path}")
            
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return {}

    @staticmethod
    def get_provider_config(config_path: str, provider: str) -> Dict[str, Any]:
        """
        获取Provider配置（支持两级路径）

        Args:
            config_path: 配置文件路径
            provider: Provider路径（如 "openai.openrouter", "vertex.official"）

        Returns:
            Provider配置字典
        """
        config = ConfigLoader.load_toml(config_path)
        providers = config.get("providers", {})
        
        # 支持两级路径
        parts = provider.split(".", 1)
        if len(parts) == 2:
            # 两级：Provider配置（如 "openai.openrouter", "vertex.official"）
            first_level = providers.get(parts[0], {})
            if isinstance(first_level, dict):
                return first_level.get(parts[1], {})
        elif len(parts) == 1:
            # 单级：直接查找（向后兼容，但不推荐）
            logger.warning(f"Provider路径 '{provider}' 只有一级，建议使用两级格式如 '{provider}.official'")
            return providers.get(parts[0], {})
        
        logger.warning(f"Provider路径格式错误，应为 'xxx.yyy' 格式，实际为: {provider}")
        return {}
    
    @staticmethod
    def get_model_config(config_path: str, model_name: str) -> Dict[str, Any]:
        """
        获取Model配置（从 models.xxx 读取）

        Args:
            config_path: 配置文件路径
            model_name: Model名称（如 "gemini-2.5pro-OB"）

        Returns:
            Model配置字典（包含 provider, model_resource 等）
        """
        config = ConfigLoader.load_toml(config_path)
        models = config.get("models", {})
        
        if not isinstance(models, dict):
            return {}
        
        return models.get(model_name, {})
    
    @staticmethod
    def get_provider_config_with_env_fallback(config_path: str, provider: str) -> Dict[str, Any]:
        """
        获取Provider配置并合并环境变量（推荐使用此方法）

        Args:
            config_path: 配置文件路径
            provider: Provider路径（如 "openai.openrouter", "vertex.official"）

        Returns:
            Provider配置字典（包含api_key, base_url等，已合并环境变量）
        """
        provider_config = ConfigLoader.get_provider_config(config_path, provider)
        
        if not provider_config:
            logger.warning(f"Provider {provider} 在配置文件中未找到")
            return {}
        
        # 如果配置中没有 api_key 或 base_url，尝试从环境变量获取
        if not provider_config.get("api_key"):
            provider_config["api_key"] = os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("VERTEX_API_KEY")
        if not provider_config.get("base_url"):
            provider_config["base_url"] = os.getenv("OPENAI_BASE_URL") or os.getenv("GOOGLE_BASE_URL") or os.getenv("VERTEX_BASE_URL")
        
        # Vertex AI 特殊处理：支持 credentials_path 和 project_id
        # 检查 provider 是否以 "vertex" 开头（支持 "vertex.official" 等）
        if provider.lower().startswith("vertex"):
            if not provider_config.get("credentials_path"):
                provider_config["credentials_path"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if not provider_config.get("project_id"):
                provider_config["project_id"] = os.getenv("GOOGLE_CLOUD_PROJECT")
            if not provider_config.get("location"):
                provider_config["location"] = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        
        # 确保 provider 字段存在
        provider_config["provider"] = provider
        
        return provider_config
    
    @staticmethod
    def get_model_config_with_provider(config_path: str, model_name: str) -> Dict[str, Any]:
        """
        获取Model配置并合并对应的Provider配置（推荐使用此方法）
        
        重要：此方法会同时读取 Model 配置和 Provider 配置，因为验证需要 Provider 的认证信息（api_key, base_url, project_id 等）

        Args:
            config_path: 配置文件路径
            model_name: Model名称（如 "gemini-2.5pro-OB"）

        Returns:
            合并后的配置字典（包含完整的 provider 配置和 model 配置，已合并环境变量）
            - Provider 配置：api_key, base_url, project_id, location, credentials_path 等
            - Model 配置：model_resource, endpoint 等
            - Model 配置会覆盖 Provider 配置中的同名字段
        """
        # 1. 获取 Model 配置（从 models.xxx 读取）
        model_config = ConfigLoader.get_model_config(config_path, model_name)
        
        if not model_config:
            logger.warning(f"Model {model_name} 在配置文件中未找到")
            return {}
        
        # 2. 获取 Model 引用的 Provider（必须要有，否则无法验证）
        provider = model_config.get("provider")
        if not provider:
            logger.warning(f"Model {model_name} 未指定 provider，无法进行验证")
            return {}
        
        # 3. 获取 Provider 配置并合并环境变量（必须要有，包含认证信息）
        provider_config = ConfigLoader.get_provider_config_with_env_fallback(config_path, provider)
        
        if not provider_config:
            logger.warning(f"Model {model_name} 引用的 Provider {provider} 未找到，无法进行验证")
            return {}
        
        # 4. 合并配置：先 Provider 配置，后 Model 配置（Model 配置可以覆盖 Provider 配置）
        merged_config = {**provider_config, **model_config}
        
        # 5. 确保 provider 字段存在（使用 Model 配置中的 provider）
        merged_config["provider"] = provider
        
        # 6. 自动添加 model_name 字段，用于 BaseValidator 自动获取
        merged_config["model_name"] = model_name
        
        return merged_config

    @staticmethod
    def get_api_key(config_path: str, provider: str) -> Optional[str]:
        """
        从配置文件获取API密钥（支持两级provider路径）

        Args:
            config_path: 配置文件路径
            provider: Provider路径（如 "openai.openrouter", "openai.official"）

        Returns:
            API密钥
        """
        provider_config = ConfigLoader.get_provider_config(config_path, provider)
        return provider_config.get("api_key")
