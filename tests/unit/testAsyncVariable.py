import asyncio
import pytest
from unittest.mock import patch
from datetime import datetime
from typing import Optional, Tuple

from src.utils.asyncVariable import AsyncVariable

# 假设 mstime() 返回时间戳（例如 datetime.now().timestamp()）
def mstime() -> float:
    return datetime.now().timestamp()

class TestAsyncVariable:
    @pytest.mark.asyncio
    async def test_set_and_get(self):
        """测试 set 和 get 基本功能"""
        av = AsyncVariable()
        assert await av.get() is None

        await av.set(42)
        assert await av.get() == 42

    @pytest.mark.asyncio
    async def test_wait_not_none_immediate(self):
        """当值已存在时，wait_not_none 立即返回"""
        av = AsyncVariable()
        await av.set("ready")
        result = await av.wait_not_none()
        assert result == "ready"

    @pytest.mark.asyncio
    async def test_wait_not_none_with_delay(self):
        """测试 wait_not_none 等待值被设置"""
        av = AsyncVariable()

        async def set_later():
            await asyncio.sleep(0.1)
            await av.set("delayed")

        # 同时启动设置值和等待任务
        await asyncio.gather(set_later(), av.wait_not_none())
        assert await av.get() == "delayed"

    @pytest.mark.asyncio
    async def test_wait_not_none_timeout(self):
        """测试 wait_not_none 超时抛出异常"""
        av = AsyncVariable()
        with pytest.raises(asyncio.TimeoutError):
            await av.wait_not_none(timeout=0.1)

    @pytest.mark.asyncio
    async def test_wait_set(self):
        """测试 wait_set 等待下一次 set 操作"""
        av = AsyncVariable()
        initial_count = av._set_count  # 访问内部变量仅为测试

        # 启动一个设置值的任务
        async def set_values():
            await asyncio.sleep(0.1)
            await av.set(1)
            await asyncio.sleep(0.1)
            await av.set(2)

        # 等待第一次 set
        await av.wait_set()
        assert av._set_count == initial_count + 1
        assert await av.get() == 1

        # 等待第二次 set
        await av.wait_set()
        assert av._set_count == initial_count + 2
        assert await av.get() == 2

    @pytest.mark.asyncio
    async def test_wait_set_timeout(self):
        """测试 wait_set 超时抛出异常"""
        av = AsyncVariable()
        with pytest.raises(asyncio.TimeoutError):
            await av.wait_set(timeout=0.1)

    @pytest.mark.asyncio
    async def test_clear(self):
        """测试 clear 方法重置值和时间戳"""
        av = AsyncVariable()
        await av.set("data")
        await av.clear()
        assert await av.get() is None
        assert await av.get_timestamp() is None

    @pytest.mark.asyncio
    async def test_timestamp(self):
        """测试时间戳相关方法"""
        av = AsyncVariable()
        before = mstime()
        await av.set("test")
        after = mstime()

        value, ts = await av.get_with_timestamp()
        assert value == "test"
        assert before <= ts <= after

        # 测试 ISO 格式转换
        iso_ts = await av.get_iso_timestamp()
        assert iso_ts == datetime.fromtimestamp(ts).isoformat()

    @pytest.mark.asyncio
    async def test_concurrent_waiters(self):
        """测试多个协程同时等待值的场景"""
        av = AsyncVariable()
        results = []

        async def waiter1():
            results.append(await av.wait_not_none())

        async def waiter2():
            results.append(await av.wait_not_none())

        async def setter():
            await asyncio.sleep(0.1)
            await av.set("done")

        # 同时启动两个等待者和一个设置者
        await asyncio.gather(waiter1(), waiter2(), setter())
        assert results == ["done", "done"]

    @pytest.mark.asyncio
    async def test_wait_with_timestamp(self):
        """测试带时间戳的等待方法"""
        av = AsyncVariable()
        
        async def set_value():
            await asyncio.sleep(0.1)
            await av.set(100)

        # 同时启动设置和等待
        task = asyncio.create_task(set_value())
        value, ts = await av.wait_with_timestamp()
        await task

        assert value == 100
        assert isinstance(ts, float)

    @pytest.mark.asyncio
    async def test_set_same_value(self):
        """测试多次设置相同值时，wait_set 仍能触发"""
        av = AsyncVariable()
        await av.set("A")
        count_before = av._set_count

        # 再次设置相同值
        await av.set("A")
        assert av._set_count == count_before + 1

        # 确保 wait_set 能触发
        await av.wait_set(timeout=0.1)