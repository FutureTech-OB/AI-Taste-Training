"""
Article workers 工具函数
"""
import asyncio
import os
import time
from typing import Optional, Union, Dict, Any, AsyncIterator, Iterator
from beanie import UpdateResponse


async def task_generator(
    doc_type: type, 
    filter: dict = {},
    set: dict = {},
    inc: dict = {},
    sleep_time: Optional[Union[int, str]] = None,
) -> AsyncIterator[Any]:
    """
    异步任务生成器 - 使用原子更新操作避免并发问题
    
    使用 find_one().update() 的原子性来确保多个 worker 不会同时处理同一个文档。
    这是通过原子更新操作实现的，即使更新操作本身没有业务意义，也能起到"锁定"文档的作用。
    
    Args:
        doc_type: Beanie Document 类型
        filter: 查询过滤器
        set: 要设置的字段（使用 Set 操作符）
        inc: 要递增的字段（使用 Inc 操作符）
        sleep_time: 没有任务时的睡眠时间（秒），默认从环境变量 SLEEP_TIME 读取，或使用 1
    
    Yields:
        找到并更新的文档对象
    """
    if sleep_time is None:
        sleep_time = os.getenv("SLEEP_TIME", "1")
    if isinstance(sleep_time, str):
        sleep_time = int(sleep_time)
    
    # 构建更新操作（使用 MongoDB 原生格式）
    update_ops = {}
    if set:
        update_ops["$set"] = set
    if inc:
        update_ops["$inc"] = inc
    
    iteration = 0
    while True:
        try:
            if update_ops:
                # 如果有更新操作，使用原子更新
                task = await doc_type.find_one(filter).update(
                    update_ops,
                    response_type=UpdateResponse.NEW_DOCUMENT
                )
            else:
                # 如果没有更新操作，直接使用 find_one
                task = await doc_type.find_one(filter)
            
            if task:
                yield task
                iteration = 0  # 重置计数器
            else:
                iteration += 1
                # 每100次迭代（约10秒）打印一次日志
                if iteration % 100 == 0:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"等待符合条件的文档... (已等待 {iteration * sleep_time:.1f} 秒)")
                await asyncio.sleep(sleep_time)
        except Exception as e:
            # 如果出错，等待一段时间后继续
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"task_generator 错误: {e}")
            await asyncio.sleep(sleep_time)


def sync_task_generator(
    doc_type: type,    
    filter: dict = None,
    set: dict = None,
    inc: dict = None,
    sleep_time: Optional[int] = None,
) -> Iterator[Any]:
    """
    同步任务生成器 - 在同步环境中使用异步操作
    
    Args:
        doc_type: Beanie Document 类型
        filter: 查询过滤器
        set: 要设置的字段（使用 Set 操作符）
        inc: 要递增的字段（使用 Inc 操作符）
        sleep_time: 没有任务时的睡眠时间（秒），默认从环境变量 SLEEP_TIME 读取，或使用 1
    
    Yields:
        找到并更新的文档对象
    
    Note:
        这个函数在内部创建新的事件循环来运行异步操作
    """
    if filter is None:
        filter = {}
    if set is None:
        set = {}
    if inc is None:
        inc = {}
    
    if sleep_time is None:
        sleep_time = int(os.getenv("SLEEP_TIME", "1"))
    
    # 构建更新操作（使用 MongoDB 原生格式）
    update_ops = {}
    if set:
        update_ops["$set"] = set
    if inc:
        update_ops["$inc"] = inc
    
    # 使用线程本地存储来管理事件循环
    loop = None
    
    def get_or_create_loop():
        """获取或创建事件循环"""
        nonlocal loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    
    while True:
        try:
            # 获取或创建事件循环
            loop = get_or_create_loop()
            
            # 运行异步操作
            if update_ops:
                task = loop.run_until_complete(
                    doc_type.find_one(filter).update(
                        update_ops,
                        response_type=UpdateResponse.NEW_DOCUMENT
                    )
                )
            else:
                task = loop.run_until_complete(
                    doc_type.find_one(filter)
                )
            
            if task:
                yield task
            else:
                time.sleep(sleep_time)
        except Exception as e:
            # 如果出错，等待一段时间后继续
            time.sleep(sleep_time)


def merge_context(*args) -> str:
    """
    合并上下文字符串
    
    Args:
        *args: 要合并的字符串参数
    
    Returns:
        合并后的字符串
    """
    return "\n".join(str(arg) for arg in args if arg)

