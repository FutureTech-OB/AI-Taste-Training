"""
Article Practice的提示词管理
"""
from typing import Optional, Dict


class ArticlePrompts:
    """Article相关的提示词"""

    _prompts: Dict[str, str] = {
        "rank_classifier": """You are a research quality classifier. Classify the research article into one of these categories:
- exceptional: High-impact journal, rigorous methodology, reliable conclusions, widely cited
- strong: Good journal, reasonable research design, meaningful contribution
- fair: Average journal, basic standards met, some limitations
- limited: Lower quality, obvious issues in methods or conclusions
- others: Does not fit above categories
- no_match: Not a research article

Respond with only the category name.""",

        "subject_classifier": """You are a subject area classifier. Identify the primary subject area of this research article.
Common subjects include: math, physics, chemistry, biology, computer_science, engineering, medicine, etc.

Respond with only the subject name in lowercase.""",
    }

    @classmethod
    def get_prompt(cls, prompt_name: str) -> Optional[str]:
        """
        获取提示词

        Args:
            prompt_name: 提示词名称

        Returns:
            提示词内容，如果不存在返回None
        """
        return cls._prompts.get(prompt_name)

    @classmethod
    def add_prompt(cls, prompt_name: str, prompt_content: str):
        """
        添加新的提示词

        Args:
            prompt_name: 提示词名称
            prompt_content: 提示词内容
        """
        cls._prompts[prompt_name] = prompt_content

    @classmethod
    def list_prompts(cls) -> list:
        """列出所有可用的提示词名称"""
        return list(cls._prompts.keys())
