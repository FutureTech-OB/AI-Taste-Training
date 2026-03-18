"""
PDF数据模型
"""
import datetime
from pydantic import BaseModel, Field
from beanie import Document, Link
from typing import Optional, List, Tuple, Dict, Any
import httpx
import logging

logger = logging.getLogger(__name__)


class PDFData(Document):
    """PDF二进制数据"""
    data: bytes


# class PDFDocument(Document):
#     """PDF文档模型"""

#     session_id: Optional[str] = None
#     title: str
#     pdf_data: Link["PDFData"]
#     callback_url: Optional[str] = None
#     parser: str = "fitz"
#     status: str = "created"
#     created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)

#     model_id: str = "glm-4.5"
#     params: dict = {}
#     trial_count: int = 0

#     content: str = ""
#     extracted_images: Optional[List[Dict[str, Any]]] = None
#     extracted_tables: Optional[List[Dict[str, Any]]] = None
#     generated_images: Optional[List[Dict[str, Any]]] = None
    
#     class Settings:
#         indexes = [
#             "session_id",
#             "trial_count",
#             "status",
#             "parser",
#         ]

#     async def save(self, callback:bool = False):
#         await super().save()
#         if callback and self.callback_url:
#             async with httpx.AsyncClient() as client:
#                 response = await client.post(
#                     self.callback_url,
#                     json={"status": self.status, "session_id": self.session_id},
#                     headers={"Content-Type": "application/json"}
#                 )
#             if response.status_code != 200:
#                 logger.error(f"Failed to send callback to {self.callback_url}")

