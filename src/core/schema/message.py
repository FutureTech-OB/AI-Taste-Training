"""
标准消息格式定义
"""
from typing import Literal
from pydantic import BaseModel, Field
from enum import Enum


class MessageRole(str, Enum):
    """消息角色"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    MODEL = "model"  # Gemini使用


class Message(BaseModel):
    """标准消息格式"""
    role: MessageRole
    content: str = Field(..., min_length=1)

    class Config:
        use_enum_values = True

    def to_dict(self):
        """转换为字典"""
        return {
            "role": self.role,
            "content": self.content
        }
