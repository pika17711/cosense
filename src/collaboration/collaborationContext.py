from __future__ import annotations

from enum import IntEnum, auto
import logging
import queue
import threading
from typing import Callable, Optional, Union

from appConfig import AppConfig
import appType

from collaboration.transactionHandler import transactionHandler

from utils.common import mstime, server_assert

class CContextCotorState(IntEnum):
    """协作对话状态枚举"""
    PENDING = auto()         # 初始状态
    SENDNTY = auto()         # 收到订阅请求，未发出通知响应
    SUBSCRIBED = auto()      # 被订阅中
    CLOSED = auto()          # 已终止

    def __str__(self):
        return "CContextCotorState." + self.name

    def handle(self, 
               pending_func: Optional[Callable[[], None]],
               sendnty_func: Optional[Callable[[], None]],
               subscribed_func: Optional[Callable[[], None]],
               closed_func: Optional[Callable[[], None]]):
        if self == CContextCotorState.PENDING:
            if pending_func is not None:
                pending_func()
        elif self == CContextCotorState.SENDNTY:
            if sendnty_func is not None:
                sendnty_func()
        elif self == CContextCotorState.SUBSCRIBED:
            if subscribed_func is not None:
                subscribed_func()
        elif self == CContextCotorState.CLOSED:
            if closed_func is not None:
                closed_func()


class CContextCoteeState(IntEnum):
    PENDING = auto()         # 初始状态
    WAITNTY = auto()         # 发出订阅请求，等待通知响应
    SUBSCRIBING = auto()     # 订阅中
    CLOSED = auto()          # 已终止

    def __str__(self):
        return "CContextCoteeState." + self.name

    def handle(self,
               pending_func: Optional[Callable[[], None]],
               waitnty_func: Optional[Callable[[], None]],
               subscribing_func: Optional[Callable[[], None]],
               closed_func: Optional[Callable[[], None]]):
        if self == CContextCoteeState.PENDING:
            if pending_func is not None:
                pending_func()
        elif self == CContextCoteeState.WAITNTY:
            if waitnty_func is not None:
                waitnty_func()
        elif self == CContextCoteeState.SUBSCRIBING:
            if subscribing_func is not None:
                subscribing_func()
        elif self == CContextCoteeState.CLOSED:
            if closed_func is not None:
                closed_func()

class BCContextState(IntEnum):
    PENDING = auto()         # 初始状态
    WAITBNTY = auto()        # 已发送广播订阅等待广播订阅通知
    CLOSED = auto()          # 已终止

    def __str__(self):
        return "BCContextState." + self.name

class CSContextCotorState(IntEnum):
    PENDING = auto()         # 初始状态
    SENDREQ = auto()         # 已经发送SENDREQ
    SENDRDY = auto()         # 已经收到SENDRDY
    SENDEND = auto()         # 已经发出SENDEND

    def __str__(self):
        return 'CSContextCotorState.' + self.name

    def handle(self,
               pending_func: Optional[Callable[[], None]],
               sendreq_func: Optional[Callable[[], None]],
               sendrdy_func: Optional[Callable[[], None]],
               sendend_func: Optional[Callable[[], None]]):
        if self == CSContextCotorState.PENDING:
            if pending_func is not None:
                pending_func()
        elif self == CSContextCotorState.SENDREQ:
            if sendreq_func is not None:
                sendreq_func()
        elif self == CSContextCotorState.SENDRDY:
            if sendrdy_func is not None:
                sendrdy_func()
        elif self == CSContextCotorState.SENDEND:
            if sendend_func is not None:
                sendend_func()

class CSContextCoteeState(IntEnum):
    PENDING = auto()         # 初始状态
    WAITRDY = auto()         # 等待RECVRDY
    RECVRDY = auto()         # 已经收到RECVRDY
    RECVEND = auto()         # 已经收到RECVEND

    def __str__(self):
        return "CSContextCotorState." + self.name

    def handle(self,
               pending_func: Optional[Callable[[], None]],
               waitrdy_func: Optional[Callable[[], None]],
               recvrdy_func: Optional[Callable[[], None]],
               recvend_func: Optional[Callable[[], None]]):
        if self == CSContextCoteeState.PENDING:
            if pending_func is not None:
                pending_func()
        elif self == CSContextCoteeState.WAITRDY:
            if waitrdy_func is not None:
                waitrdy_func()
        elif self == CSContextCoteeState.RECVRDY:
            if recvrdy_func is not None:
                recvrdy_func()
        elif self == CSContextCoteeState.RECVEND:
            if recvend_func is not None:
                recvend_func()

"""
CollaborationContext
"""
class CContext:
    def __init__(self, 
                 cfg: AppConfig,
                 cid: appType.cid_t,
                 cotor: appType.id_t,
                 cotee: appType.id_t,
                 ):
        self.cfg = cfg
        self.cid: appType.cid_t = cid                  # 对话ID
        self.cotor: appType.id_t = cotor               # 协作者ID
        self.cotee: appType.id_t = cotee               # 被协作者ID

        self.state: Union[CContextCoteeState, CContextCotorState] = CContextCotorState.PENDING if self.is_cotor() else CContextCoteeState.PENDING

        self.last_active = mstime()                      # 最后活跃时间
        self.start_time = mstime()                       # 开始时间

        self.lock = threading.RLock()
        self.sid_set_event = threading.Event()
        self.stream_state: Union[CSContextCoteeState, CSContextCotorState] = CSContextCotorState.PENDING if self.is_cotor() else CSContextCoteeState.PENDING
        self.sid: Optional[appType.sid_t] = None


    def is_cotor(self) -> bool:
        return self.cotor == self.cfg.id

    def is_cotee(self) -> bool:
        return not self.is_cotor()

    def is_expired(self) -> bool:
        # 用上一次活跃的时间检查是否存活
        return (mstime() - self.last_active) > self.cfg.cctx_keepalive

    def local_id(self) -> appType.id_t:
        return self.cotee if self.is_cotee() else self.cotor

    def remote_id(self) -> appType.id_t:
        return self.cotee if self.is_cotor() else self.cotor

    def update_active(self):
        self.last_active = mstime()

    def is_alive(self) -> bool:
        return self.state not in (CContextCoteeState.CLOSED, CContextCotorState.CLOSED)

    def have_sid(self) -> bool:
        return self.sid is not None
    
    def __str__(self) -> str:
        return f"context: {self.cid}, remote: {self.remote_id()}, state: {self.state}"