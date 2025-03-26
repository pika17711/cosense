import asyncio
import datetime
from typing import Optional, Any, Tuple

import numpy as np


def mstime() -> float:
    return datetime.datetime.now().timestamp()


def calculate_overlap(ego_mask: np.ndarray, mask: np.ndarray):
    c = 1
    for s in ego_mask.shape:
        c *= s

    return np.count_nonzero(ego_mask * mask) / c

class AsyncVariable:
    def __init__(self):
        self._condition = asyncio.Condition()
        self._value: Optional[Any] = None
        self._timestamp: Optional[float] = None  # 新增时间戳字段

    async def set(self, value: Any) -> None:
        """设置值并记录当前时间戳"""
        async with self._condition:
            self._value = value
            self._timestamp = mstime()  # 自动生成时间戳
            self._condition.notify_all()

    async def get(self) -> Optional[Any]:
        """立即获取当前值（可能为None）"""
        return self._value

    async def wait(self, timeout: Optional[float] = None) -> Any:
        """异步等待值被设置，支持超时"""
        if self._value is not None:
            return self._value
        async with self._condition:
            # 等待条件满足
            await self._wait_condition(timeout)
            if self._value is None:
                raise asyncio.TimeoutError("等待超时")
            return self._value

    async def get_with_timestamp(self) -> Tuple[Optional[Any], Optional[float]]:
        """获取值和时间戳的元组"""
        return self._value, self._timestamp

    async def wait_with_timestamp(self, timeout: Optional[float] = None) -> Tuple[Any, float]:
        """等待并返回值和对应的时间戳"""
        if self._value is not None:
            return self._value, self._timestamp
        async with self._condition:
            await self._wait_condition(timeout)
            if self._value is None or self._timestamp is None:
                raise asyncio.TimeoutError("等待超时")
            return self._value, self._timestamp

    async def _wait_condition(self, timeout: Optional[float] = None) -> None:
        """封装等待逻辑"""
        if timeout is not None:
            await asyncio.wait_for(
                self._condition.wait_for(lambda: self._value is not None),
                timeout
            )
        else:
            await self._condition.wait_for(lambda: self._value is not None)

    async def clear(self) -> None:
        """重置变量和时间戳"""
        async with self._condition:
            self._value = None
            self._timestamp = None

    async def get_timestamp(self) -> Optional[float]:
        """单独获取时间戳"""
        return self._timestamp

    async def get_iso_timestamp(self) -> Optional[str]:
        """获取ISO格式的时间戳字符串"""
        return self._timestamp.isoformat() if self._timestamp else None