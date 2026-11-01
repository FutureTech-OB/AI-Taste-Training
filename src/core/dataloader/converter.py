"""
数据格式转换器 - 将数据流转换为训练文件格式
"""
import json
import aiofiles
from typing import AsyncIterator, Dict, Any, List
from pathlib import Path


class DataConverter:
    """数据格式转换器"""

    @staticmethod
    async def to_jsonl(
        data_stream: AsyncIterator[Dict[str, Any]],
        output_path: str
    ) -> str:
        """
        流式写入JSONL（边读边写，不占内存）

        Args:
            data_stream: 数据流
            output_path: 输出文件路径

        Returns:
            输出文件路径
        """
        # 确保目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
            async for item in data_stream:
                line = json.dumps(item, ensure_ascii=False)
                await f.write(line + "\n")

        return output_path

    @staticmethod
    async def to_parquet(
        data_stream: AsyncIterator[Dict[str, Any]],
        output_path: str,
        batch_size: int = 10000
    ) -> str:
        """
        批量写入Parquet（分批写入，控制内存）

        Args:
            data_stream: 数据流
            output_path: 输出文件路径
            batch_size: 每批写入的记录数

        Returns:
            输出文件路径
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        # 确保目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        writer = None
        batch = []

        try:
            async for item in data_stream:
                batch.append(item)

                if len(batch) >= batch_size:
                    table = pa.Table.from_pylist(batch)
                    if writer is None:
                        writer = pq.ParquetWriter(output_path, table.schema)
                    writer.write_table(table)
                    batch = []

            # 写入最后一批
            if batch:
                table = pa.Table.from_pylist(batch)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema)
                writer.write_table(table)

        finally:
            if writer:
                writer.close()

        return output_path
