import asyncio
from typing import Any, Tuple, Optional

from src.utils.common import mstime

class AsyncVariable:
    def __init__(self):
        self._condition = asyncio.Condition()
        self._value: Optional[Any] = None
        self._timestamp: Optional[float] = None
        self._set_count = 0  # 用于跟踪 set 操作的计数器

    async def set(self, value: Any) -> None:
        """设置值并记录时间戳，触发所有等待条件"""
        async with self._condition:
            self._value = value
            self._timestamp = mstime()
            self._set_count += 1  # 每次 set 操作增加计数器
            self._condition.notify_all()

    async def get(self) -> Optional[Any]:
        """立即获取当前值（可能为 None）"""
        return self._value

    async def wait_not_none(self, timeout: Optional[float] = None) -> Any:
        """等待直到值非 None（若已非 None 则立即返回）"""
        if self._value is not None:
            return self._value
        async with self._condition:
            await self._wait_condition(lambda: self._value is not None, timeout)
            if self._value is None:
                raise asyncio.TimeoutError("等待超时")
            return self._value

    async def wait_set(self, timeout: Optional[float] = None) -> Any:
        """等待下一次 set 操作（无论当前值是否为 None）"""
        current_count = self._set_count
        async with self._condition:
            await self._wait_condition(
                lambda: self._set_count != current_count, timeout
            )
            return self._value

    async def get_with_timestamp(self) -> Tuple[Optional[Any], Optional[float]]:
        """获取值和时间戳的元组"""
        return self._value, self._timestamp

    async def wait_with_timestamp(self, timeout: Optional[float] = None) -> Tuple[Any, float]:
        """等待并返回非 None 值及其时间戳"""
        if self._value is not None:
            return self._value, self._timestamp
        async with self._condition:
            await self._wait_condition(lambda: self._value is not None, timeout)
            return self._value, self._timestamp

    async def _wait_condition(self, predicate, timeout: Optional[float] = None) -> None:
        """通用等待条件封装"""
        if timeout is not None:
            await asyncio.wait_for(
                self._condition.wait_for(predicate),
                timeout
            )
        else:
            await self._condition.wait_for(predicate)

    async def clear(self) -> None:
        """重置值和时间戳"""
        async with self._condition:
            self._value = None
            self._timestamp = None

    async def get_timestamp(self) -> Optional[float]:
        """单独获取时间戳"""
        return self._timestamp

    async def get_iso_timestamp(self) -> Optional[str]:
        """获取 ISO 格式的时间戳字符串"""
        return self._timestamp.isoformat() if self._timestamp else None