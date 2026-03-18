"""
文件名清理工具
"""
import re
from pathlib import Path


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    清理文件名，移除非法字符

    Args:
        filename: 原始文件名
        max_length: 最大长度

    Returns:
        清理后的文件名
    """
    # 移除非法字符
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

    # 移除控制字符
    filename = re.sub(r'[\x00-\x1f\x7f]', '', filename)

    # 移除首尾空格和点
    filename = filename.strip('. ')

    # 限制长度
    if len(filename) > max_length:
        name, ext = Path(filename).stem, Path(filename).suffix
        max_name_length = max_length - len(ext)
        filename = name[:max_name_length] + ext

    return filename or "unnamed"


def sanitize_name(value: str) -> str:
    """
    Replace characters that are invalid for Windows filenames.
    This is a simpler version that only handles basic invalid characters.

    Args:
        value: 原始字符串

    Returns:
        清理后的字符串
    """
    if not value:
        return value
    invalid = '<>:"/\\|?*'
    sanitized = value
    for ch in invalid:
        sanitized = sanitized.replace(ch, "_")
    return sanitized


def sanitize_path(path: str) -> str:
    """
    清理路径，确保安全

    Args:
        path: 原始路径

    Returns:
        清理后的路径
    """
    # 规范化路径
    path = Path(path).as_posix()

    # 移除危险的路径遍历
    path = path.replace('..', '')

    return path
