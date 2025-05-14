from __future__ import annotations

from enum import IntEnum, auto
import queue
import AppType
from config import AppConfig
from utils.common import mstime

class BCContextState(IntEnum):
    PENDING = auto()         # 初始状态
    WAITBNTY = auto()        # 已发送广播订阅等待广播订阅通知
    CLOSED = auto()          # 已终止

    def __str__(self):
        return "BCContextState." + self.name

class BCContext:
    def __init__(self, 
                 cfg: AppConfig,
                 cid: AppType.cid_t,
                 ):
        self.cfg = cfg
        self.cid = cid
        self.cctx_set = set()
        self.start_time = mstime()
        self.state = BCContextState.PENDING
        self.msg_queue = queue.Queue()

    def is_expired(self) -> bool:
        # 用开始时间计算是否存活
        return mstime() - self.start_time > self.cfg.bcctx_keepalive

    def is_alive(self) -> bool:
        return self.state != BCContextState.CLOSED
