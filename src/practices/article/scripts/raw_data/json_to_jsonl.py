"""
将 JSON 数组文件转为 JSONL（一行一个 JSON 对象）。
大文件用 ijson 流式解析，避免 OOM。
"""
import json
import logging
from pathlib import Path
from typing import Any, Union

logger = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    """让 MongoDB 导出的 Decimal/ObjectId 等可被 json.dumps 序列化。"""
    if hasattr(obj, "__float__"):  # Decimal
        return float(obj)
    if hasattr(obj, "__str__") and "oid" in type(obj).__name__.lower():
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def json_array_to_jsonl(
    input_path: Union[str, Path],
    output_path: Union[str, Path, None] = None,
    *,
    stream: bool = True,
) -> Path:
    """
    把 .json 里的 JSON 数组写成 .jsonl，每行一个对象。

    Args:
        input_path: 输入的 .json 文件
        output_path: 输出的 .jsonl 文件；默认与输入同目录，扩展名改为 .jsonl
        stream: 若 True 且已安装 ijson，则流式解析（适合大文件）；否则整文件 json.load

    Returns:
        输出文件路径
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".jsonl")
    else:
        output_path = Path(output_path)

    if stream:
        try:
            import ijson
        except ImportError:
            logger.warning("ijson 未安装，改用 json.load；大文件可能占满内存。pip install ijson")
            stream = False

    if stream:
        _stream_json_array_to_jsonl(input_path, output_path)
    else:
        _load_json_array_to_jsonl(input_path, output_path)

    logger.info("已写入 %s", output_path)
    return output_path


def _stream_json_array_to_jsonl(input_path: Path, output_path: Path) -> None:
    import ijson

    count = 0
    with open(input_path, "rb") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for obj in ijson.items(fin, "item"):
            fout.write(
                json.dumps(obj, ensure_ascii=False, default=_json_default) + "\n"
            )
            count += 1
            if count % 50000 == 0:
                logger.info("已写出 %d 条", count)
    logger.info("共写出 %d 条", count)


def _load_json_array_to_jsonl(input_path: Path, output_path: Path) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"输入不是 JSON 数组: {input_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for i, obj in enumerate(data):
            f.write(
                json.dumps(obj, ensure_ascii=False, default=_json_default) + "\n"
            )
            if (i + 1) % 50000 == 0:
                logger.info("已写出 %d 条", i + 1)
    logger.info("共写出 %d 条", len(data))
