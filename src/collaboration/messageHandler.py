import asyncio
import logging
import random
from typing import Any, Callable, Coroutine, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from collaboration.message import AppRegMessage, AppRspMessage, NotifyAct, RecvEndMessage, RecvFileMessage, RecvMessage, RecvRdyMessage, SendEndMessage, SendFileMessage, SendFinMessage, SendRdyMessage, SendReqMessage, SubscribeAct
from collaboration.transactionHandler import transactionHandler
from utils import InfoDTO
from config import AppConfig

from enum import IntEnum, auto

from utils.common import load_json, mstime, read_binary_file, server_assert, server_logic_error, server_not_implemented, string_to_32_hex, sync_to_async

from collaboration.messageID import MessageID
from cachetools import LRUCache

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
    def __init__(self, cid: AppConfig.cid_t, cotor, cotee, bcctx, mh: 'MessageHandler'):
        self.cid: AppConfig.cid_t = cid                  # 对话ID
        self.cotor: AppConfig.id_t = cotor               # 协作者ID
        self.cotee: AppConfig.id_t = cotee               # 被协作者ID
        self.state = CContextCotorState.PENDING if self.is_cotor() else CContextCoteeState.PENDING
        self.bcctx: BCContext = bcctx                    # 广播协作对话
        self.last_active = mstime()                      # 最后活跃时间
        self.start_time = mstime()                       # 开始时间
        self.msg_queue: asyncio.Queue[Message] = asyncio.Queue()
        self.message_handler: MessageHandler = mh
        self.sid_set_event = asyncio.Event()
        self.stream_state = CSContextCotorState.PENDING
        self.sid: Optional[AppConfig.sid_t] = None

    def is_cotor(self) -> bool:
        return self.cotor == AppConfig.id

    def is_cotee(self) -> bool:
        return not self.is_cotor()

    def is_expired(self) -> bool:
        # 用上一次活跃的时间检查是否存活
        return (mstime() - self.last_active) > AppConfig.cctx_keepalive

    def local_id(self) -> AppConfig.id_t:
        return self.cotee if self.is_cotee() else self.cotor

    def remote_id(self) -> AppConfig.id_t:
        return self.cotee if self.is_cotor() else self.cotor

    def update_active(self):
        self.last_active = mstime()

    def is_alive(self) -> bool:
        return self.state not in (CContextCoteeState.CLOSED, CContextCotorState.CLOSED)

    def have_sid(self) -> bool:
        return self.sid is not None
    
    def to_waitnty(self):
        server_assert(self.is_cotee(), "上下文角色必须是被协作者")
        server_assert(self.state == CContextCoteeState.PENDING)
        logging.debug(f"context: {self.cid}, 状态 {self.state} -> CContextCoteeState.WAITNTY")
        self.update_active()
        self.state = CContextCoteeState.WAITNTY
        self.message_handler.add_waitnty(self)

    def to_subscribing(self):
        server_assert(self.is_cotee(), "上下文角色必须是被协作者")
        server_assert(self.state == CContextCoteeState.WAITNTY)
        logging.debug(f"context: {self.cid}, 状态 {self.state} -> CContextCoteeState.SUBSCRIBING")
        self.update_active()
        self.message_handler.rem_waitnty(self)
        logging.debug(f"订阅 {self.remote_id()}, context: {self.cid}")
        self.state = CContextCoteeState.SUBSCRIBING
        self.message_handler.add_subscribing(self)

    def to_sendnty(self):
        server_assert(self.is_cotor(), "上下文角色必须是协作者")
        server_assert(self.state == CContextCotorState.PENDING)
        logging.debug(f"context: {self.cid}, 状态 {self.state} -> CContextCotorState.SENDNTY")
        self.update_active()
        self.state = CContextCotorState.SENDNTY
        self.message_handler.add_sendnty(self)

    def to_subscribed(self):
        server_assert(self.is_cotor(), "上下文角色必须是协作者")
        server_assert(self.state == CContextCotorState.SENDNTY)
        logging.debug(f"context: {self.cid}, 状态 {self.state} -> CContextCotorState.SUBSCRIBED")
        self.update_active()
        self.message_handler.rem_sendnty(self)
        logging.debug(f"被 { self.remote_id()} 订阅, context: {self.cid}")
        self.state = CContextCotorState.SUBSCRIBED
        self.message_handler.add_subscribed(self)

    def to_closed(self):
        self.update_active()
        if self.is_cotor():
            logging.debug(f"context: {self.cid}, 状态 {self.state} -> CContextCotorState.CLOSED")
            if self.state == CContextCotorState.SUBSCRIBED:
                self.message_handler.rem_subscribed(self)
            self.state = CContextCotorState.CLOSED
        else:
            logging.debug(f"context: {self.cid}, 状态 {self.state} -> CContextCoteeState.CLOSED")            
            if self.state == CContextCoteeState.SUBSCRIBING:
                self.message_handler.rem_subscribing(self)
            self.state = CContextCoteeState.CLOSED

        self.stream_to_end()
        self.message_handler.rem_cctx(self)

    def force_close(self):
        """
            强行关闭
            无论当前状态是什么
            与to_close不同的是, 自动发送关闭会话的消息
        """
        if self.is_cotor():
            if self.state == CContextCotorState.SUBSCRIBED:
                self.message_handler.rem_subscribed(self)
                asyncio.create_task(self.message_handler.subscribe_send(self.remote_id(), SubscribeAct.FIN))
        else:
            if self.state == CContextCoteeState.SUBSCRIBING:
                self.message_handler.rem_subscribing(self)
                asyncio.create_task(self.message_handler.notify_send(self.remote_id(), NotifyAct.FIN))

        self.state = CContextCoteeState.CLOSED if self.is_cotee() else CContextCotorState.CLOSED
        self.stream_force_to_end()
        self.message_handler.rem_cctx(self)

    async def stream_get(self):
        server_assert(self.is_cotor())
        if self.stream_state == CSContextCotorState.PENDING:
            logging.debug(f"context: {self.cid} 获取stream")
            rl = 1
            pt = 1
            aoi = 0
            mode = 1
            await self.message_handler.sendreq_send(self.remote_id(), self.cid, rl, pt, aoi, mode)
            self.stream_to_sendreq()
            await self.sid_set_event.wait()
            logging.debug(f"context: {self.cid} 获得stream {self.sid}")
            server_assert(self.stream_state == CSContextCotorState.SENDRDY)

    def stream_to_waitrdy(self):
        server_assert(self.is_cotee())
        self.update_active()
        self.stream_state = CSContextCoteeState.WAITRDY

    def stream_to_recvrdy(self):
        server_assert(self.is_cotee())
        self.update_active()
        self.stream_state = CSContextCoteeState.RECVRDY

    def stream_to_sendreq(self):
        server_assert(self.is_cotor())
        self.update_active()
        self.stream_state = CSContextCotorState.SENDREQ

    def stream_to_sendrdy(self):
        server_assert(self.is_cotor())
        self.update_active()
        self.stream_state = CSContextCotorState.SENDRDY

    def stream_to_end(self):
        if self.is_cotor():
            self.state = CSContextCotorState.SENDEND
        else:
            self.state = CSContextCoteeState.RECVEND
        if self.have_sid():
            self.message_handler.rem_stream(self.sid)

    def stream_force_to_end(self):
        logging.debug(f"context:{self.cid} stream强行关闭")
        if self.is_cotor():
            if self.state == CSContextCotorState.SENDRDY:
                if self.have_sid():
                    asyncio.create_task(self.message_handler.sendend_send(self.sid))
                    self.message_handler.rem_stream(self.sid)
            self.state = CSContextCotorState.SENDEND
        else:
            if self.state == CSContextCoteeState.RECVRDY:
                if self.have_sid():
                    self.message_handler.rem_stream(self.sid)
            self.state = CSContextCoteeState.RECVEND

    async def send_data(self, data):
        server_assert(self.is_cotor(), "发送数据者应该是协助者")
        if not self.have_sid():
            await self.stream_get()
        if self.have_sid():
            await self.message_handler.send_send(self.sid, data)
        else:
            logging.warning(f"context:{self.cid} 未获取到stream, 发送数据失败")
"""
BroadcastCollaborationContext
"""
class BCContext:
    def __init__(self, cid, mh):
        self.cid = cid
        self.set = set()
        self.start_time = mstime()
        self.state = BCContextState.PENDING
        self.msg_queue = asyncio.Queue()
        self.message_handler: MessageHandler = mh

    def add_cctx(self, cctx: CContext):
        self.set.add(cctx)
    
    def rem_cctx(self, cctx: CContext):
        self.set.remove(cctx)

    def is_expired(self) -> bool:
        # 用开始时间计算是否存活
        return mstime() - self.start_time > AppConfig.bcctx_keepalive

    def is_alive(self) -> bool:
        return self.state != BCContextState.CLOSED

    def to_waitbnnty(self):
        server_assert(self.state == BCContextState.PENDING, "必须发送了广播订阅后，才能等待广播通知消息")
        self.state = BCContextState.WAITBNTY

    def to_close(self):
        if self.state == BCContextState.PENDING:
            logging.warning(f"广播会话 {self.cid}未发送广播订阅消息即被关闭")

        self.message_handler.rem_bcctx(self)
        self.state = BCContextState.CLOSED

    def force_close(self):
        self.message_handler.rem_bcctx(self)
        self.state = BCContextState.CLOSED

from collaboration.message import BroadcastPubMessage, Message, NotifyMessage, SendMessage, SubscribeMessage
from collaboration.message import BroadcastSubNtyMessage, BroadcastSubMessage
from config import AppConfig

class MessageHandler:
    def __init__(self, perception_client):
        self.msg_queue = asyncio.Queue()
        self.tx_handler = transactionHandler(self.msg_queue)
        self.perception_client = perception_client

        self.running = True

        # (cid, cotor, cotee) -> cctx
        self.cctx_table: Dict[Tuple[AppConfig.cid_t, AppConfig.id_t, AppConfig.id_t], CContext] = dict()
        # cid -> bcctx
        self.bcctx_table: Dict[AppConfig.cid_t, BCContext] = dict()
        # sid -> cctx
        self.stream_table: Dict[AppConfig.sid_t, CContext] = dict()

        # cotee id -> cctx 正在sendnty状态的cctx, 因为只有当前是cotor的时候状态才可能为sendnty，用cotee id做为索引
        self.sendnty_table: Dict[AppConfig.id_t, CContext] = dict()
        # cotee id -> cctx 正在被订阅的cctx
        self.subscribed_table: Dict[AppConfig.id_t, CContext] = dict()

        # cotor id -> cctx 正在waitnty状态的cctx, 因为只有当前是cotee的时候状态才可能为waitnty，用cotor id做为索引
        self.waitnty_table: Dict[AppConfig.id_t, CContext] = dict()
        # cotor id -> cctx 正在订阅的cctx
        self.subscribing_table: Dict[AppConfig.id_t, CContext] = dict()

        self.cid_counter = random.randint(1, 1 << 20)

        self.data_cache = LRUCache(AppConfig.data_cache_size)  # 他车数据的缓存

    def force_close(self):
        self.tx_handler.force_close()
        self.running = False

    async def recv_loop(self):
        t1 = asyncio.create_task(self.tx_handler.recv_loop())
        t2 = asyncio.create_task(self.dispatch())
        asyncio.gather(t1, t2)

    async def dispatch(self):
        while self.running:
            msg = await self.msg_queue.get()
            await self.dispatch_message(msg)

    async def get_self_feat(self):
        ts, feat = await sync_to_async(self.perception_client.get_my_feature)()
        if ts == -1:
            assert False
        return np.array([1, 2, 3])

    async def get_my_conf_map(self):
        # ts, mp = await sync_to_async(self.perception_client.get_my_feature)()
        # if ts == -1:
            # assert False
        return np.array([1, 2, 3]).tobytes()

    async def dispatch_message(self, msg: Message):
        mid = msg.header.mid
        if mid == MessageID.APPREG:
            await self.appreg_recv(msg)
        elif mid == MessageID.APPRSP:
            await self.apprsp_recv(msg)
        elif mid == MessageID.BROCASTPUB:
            await self.broadcastpub_recv(msg)
        elif mid == MessageID.BROCASTSUB:
            await self.broadcastsub_recv(msg)
        elif mid == MessageID.BROCASTSUBNTY:
            await self.broadcastsubnty_recv(msg)
        elif mid == MessageID.SUBSCRIBE:
            await self.subscribe_recv(msg)
        elif mid == MessageID.NOTIFY:
            await self.notify_recv(msg)
        elif mid == MessageID.SENDREQ:
            await self.sendreq_recv(msg)
        elif mid == MessageID.SENDRDY:
            await self.sendrdy_recv(msg)
        elif mid == MessageID.RECVRDY:
            await self.recvrdy_recv(msg)
        elif mid == MessageID.SEND:
            await self.send_recv(msg)
        elif mid == MessageID.RECV:
            await self.recv_recv(msg)
        elif mid == MessageID.SENDEND:
            await self.sendend_recv(msg)
        elif mid == MessageID.RECVEND:
            await self.recvend_recv(msg)
        elif mid == MessageID.SENDFILE:
            await self.sendfile_recv(msg)
        elif mid == MessageID.SENDFIN:
            await self.sendfin_recv(msg)
        elif mid == MessageID.RECVFILE:
            await self.recvfile_recv(msg)
        else:
            logging.warning(f"Unhandled message type: {MessageID.get_name(mid.value)}")

    # ============== cctx ====================#
    def add_cctx(self, cctx: CContext):
        self.cctx_table[(cctx.cid, cctx.cotor, cctx.cotee)] = cctx

    def check_cctx_exist(self, cid, cotor, cotee):
        return self.get_cctx(cid, cotor, cotee) is not None

    def get_cctx(self, cid, cotor, cotee) -> Optional[CContext]:
        ctable = self.cctx_table
        t = (cid, cotor, cotee)
        if t in ctable:
            cctx = ctable[t]
        else:
            cctx = None
        return cctx

    def get_cctx_or_panic(self, cid, cotor, cotee) -> CContext:
        """
            得到cctx，若不存在，则panic
        """
        ctable = self.cctx_table
        t = (cid, cotor, cotee)
        if t in ctable:
            cctx = ctable[t]
        else:
            assert False
        return cctx

    def rem_cctx(self, cctx: CContext):
        self.cctx_table.pop((cctx.cid, cctx.cotor, cctx.cotee))
        bcctx = cctx.bcctx
        if bcctx is not None:
            bcctx.rem_cctx(cctx)
    
    async def get_cctx_and_put(self, cid, cotor, cotee, msg: Message):
        cctx = self.get_cctx(cid, cotor, cotee)
        if cctx is None:
            return
        await cctx.msg_queue.put(msg)

    def get_cctx_from_stream(self, sid: AppConfig.sid_t) -> Optional[CContext]:
        stream_table = self.stream_table
        if sid in stream_table:
            return stream_table[sid]
        else:
            return None

    def add_stream(self, sid, cctx):
        self.stream_table[sid] = cctx

    def rem_stream(self, sid):
        self.stream_table.pop(sid)

    # ============== bcctx ====================#
    def add_bcctx(self, bcctx: BCContext):
        self.bcctx_table[bcctx.cid] = bcctx

    def get_bcctx(self, cid: AppConfig.cid_t) -> Optional[BCContext]:
        ctable = self.bcctx_table
        try:
            cctx_set = ctable[cid]
        except KeyError:
            cctx_set = None

        return cctx_set

    def rem_bcctx(self, bcctx: BCContext):
        self.bcctx_table.pop(bcctx.cid)

    # ============== cotee state table ====================#
    def add_waitnty(self, cctx: CContext):
        self.waitnty_table[cctx.cotor] = cctx

    def rem_waitnty(self, cctx: CContext):
        server_assert(cctx.cotor in self.waitnty_table)
        self.waitnty_table.pop(cctx.cotor)

    def get_waitnty_by_id(self, cotor_id):
        return self.waitnty_table[cotor_id] if cotor_id in self.waitnty_table else None

    def add_subscribing(self, cctx: CContext):
        self.subscribing_table[cctx.cotor] = cctx

    def rem_subscribing(self, cctx: CContext):
        server_assert(cctx.cotor in self.waitnty_table)
        self.subscribing_table.pop(cctx.cotor)

    def get_subscribing_by_id(self, cotor_id) -> Optional[CContext]:
        return self.subscribing_table[cotor_id] if cotor_id in self.subscribing_table else None

    def get_subscribing(self) -> Iterable[CContext]:
        return self.subscribing_table.values()

    # ============== cotor state table ====================#
    def add_sendnty(self, cctx: CContext):
        self.sendnty_table[cctx.cotee] = cctx

    def rem_sendnty(self, cctx: CContext):
        server_assert(cctx.cotee in self.sendnty_table)
        self.sendnty_table.pop(cctx.cotee)

    def get_sendnty_by_id(self, cotee_id):
        return self.sendnty_table[cotee_id] if cotee_id in self.sendnty_table else None

    def add_subscribed(self, cctx: CContext):
        self.subscribed_table[cctx.cotee] = cctx

    def rem_subscribed(self, cctx: CContext):
        server_assert(cctx.cotee in self.subscribed_table)
        self.subscribed_table.pop(cctx.cotee)

    def get_subscribed(self) -> Iterable[CContext]:
        return self.subscribed_table.values()
    
    def get_subscribed_by_id(self, cotor_id):
        return self.subscribing_table[cotor_id] if cotor_id in self.subscribing_table else None

    # ============== utils ====================#
    async def wait_with_timeout(self, coro, timeout: int):
        try:
            return await asyncio.wait_for(coro, timeout=timeout/1000)
        except asyncio.TimeoutError:
            return None

    async def check_need_broadcastpub(self, msg: BroadcastPubMessage):
        """
            是否需要别人的广播推送
            1. 如果当前已经建立了local被协作的连接，则不需要再次由广播推送建立连接
                local: cotee
                remote: cotor
            2. 判断协作图的重叠是否超过阈值
                TODO
        """
        if self.get_waitnty_by_id(msg.oid) is not None or self.get_subscribing_by_id(msg.oid) is not None:
            return False
        return True

    async def check_need(self, bcctx: BCContext, msg: Message):
        # 检查是否需要的逻辑
        # TODO
        return True

    async def check_need_subscribe(self, msg: SubscribeMessage):
        return True

    async def check_need_broadcastsub(self, msg: BroadcastSubMessage):
        """
            是否需要别人的广播订阅
            1. 如果当前已经建立了local协作的连接，则不需要再次由广播推送建立连接
                local: cotor
                remote: cotee
            2. 判断协作图的重叠是否超过阈值
                TODO
        """
        if self.get_sendnty_by_id(msg.oid) is not None or self.get_subscribed_by_id(msg.oid):
            return False

        return True
    
    def cid_gen(self) -> AppConfig.cid_t:
        self.cid_counter += 1
        return string_to_32_hex(str(AppConfig.app_id) + str(mstime()) + str(self.cid_counter))

    # ============== ctx loop ====================#
    async def ctx_loop(self, ctx: Union[CContext, BCContext], 
                       handler_table: Dict[MessageID, Callable[[Message], Coroutine[Any, Any, None]]]):
        while ctx.is_alive():
            msg = await self.wait_with_timeout(ctx.msg_queue.get(), 100) # 随便取的100ms，只会影响超时的准确性
            if msg is not None and msg.header.mid in handler_table:
                await handler_table[msg.header.mid](msg)
            if ctx.is_expired():
                ctx.force_close()

    async def bcctx_loop(self, bcctx: BCContext):
        handler_table = {MessageID.BROCASTSUB: self.broadcastsub_service,
                         MessageID.BROCASTSUBNTY: self.broadcastsubnty_service}
        await self.ctx_loop(bcctx, handler_table)

    async def cctx_loop(self, cctx: CContext):
        handler_table = {MessageID.SUBSCRIBE: self.subscribe_service,
                         MessageID.NOTIFY: self.notify_service,
                         MessageID.SENDRDY: self.sendrdy_service,
                         MessageID.RECVRDY: self.recvrdy_service,
                         MessageID.RECV: self.recv_service,
                         MessageID.RECVEND: self.recvend_service,}
        await self.ctx_loop(cctx, handler_table)

    # ============== 发消息逻辑 ====================#
    async def broadcastsub_send(self):
        """
            广播订阅
            描述：local开启一个新广播会话bcctx
                 启动一个新协程bcctx_loop(bcctx)，进行消息接收和处理
        """
        cid = self.cid_gen()
        coopMap = await self.get_my_conf_map()
        coopMapType = 1
        bearCap = 1
        bcctx = BCContext(cid, self)
        self.add_bcctx(bcctx)
        await self.tx_handler.brocastsub(AppConfig.id, AppConfig.topic, cid, coopMap, coopMapType, bearCap)
        bcctx.to_waitbnnty()
        asyncio.create_task(self.bcctx_loop(bcctx))

    async def broadcastpub_send(self):
        """
            广播推送
            描述：只需发送广播推送
        """
        coopMap = await self.get_my_conf_map()
        coopMapType = 1
        await self.tx_handler.brocastpub(AppConfig.id, AppConfig.topic, coopMap, coopMapType)

    async def subscribe_send(self, did: Union[List[AppConfig.id_t], AppConfig.id_t], act:SubscribeAct=SubscribeAct.ACKUPD):
        """
            订阅
            描述：
                1. act = SubscribeAct.ACKUPD
                    订阅请求
                    local开启一个新会话cctx
                    启动一个新协程cctx_loop(cctx)，进行消息接收和处理
                2. act = SubscribeAct.FIN
                    订阅关闭
                    local是cotee，remote是cotor，不然就是代码逻辑错误

            参数：
                1. did
                    目标id列表或目标id
                2. act
                    订阅请求或订阅关闭
        """
        if type(did) is AppConfig.id_t:
            did = [did]
        for didi in did:
            if act == SubscribeAct.ACKUPD:
                cid = self.cid_gen()
                cctx = CContext(cid, didi, AppConfig.id, None, self)
                self.add_cctx(cctx)
                coopMap = await self.get_my_conf_map()
                coopMapType = 1
                bearCap = 1
                await self.tx_handler.subscribe(AppConfig.id, [didi], AppConfig.topic, act, cid, coopMap, coopMapType, bearCap)
                cctx.to_waitnty()
                asyncio.create_task(self.cctx_loop(cctx))
            else:
                cctx = self.get_subscribing_by_id(didi)
                if cctx is None:
                    logging.warning(f"试图关闭不存在的订阅关系 {cctx}")
                    return
                fakecoopmap = np.array([1]).tobytes()
                coopMapType = 1
                bearCap = 0
                await self.tx_handler.subscribe(AppConfig.id, [didi], AppConfig.topic, act, cctx.cid, fakecoopmap, coopMapType, bearCap)  # TODO 关闭订阅不需要传协作图
                cctx.to_closed()
                self.rem_cctx(cctx)

    async def notify_send(self, cid, did, act=NotifyAct.ACK):
        """
            通知
            描述：
                1. act = NotifyAct.ACK
                    确认订购
                    对话cctx在收到订阅消息时新建
                    找到对应cctx，发送确认订购后，改变cctx状态
                2. act = NotifyAct.NTY
                    对话内通知
                    未实现，没理解作用
                2. act = NotifyAct.FIN
                    取消订购
                    对话cctx在收到订阅消息时新建
                    找到对应cctx，发送取消订购后，改变cctx状态
            参数：
                1. did
                2. act
        """
        cctx = self.get_cctx_or_panic(cid, AppConfig.id, did)
        if act == NotifyAct.ACK:
            coopMap = await self.get_my_conf_map()
            coopMapType = 1
            bearCap = 1
            await self.tx_handler.notify(AppConfig.id, did, AppConfig.topic, act, cid, coopMap, coopMapType, bearCap)
        elif act == NotifyAct.NTY:
            pass
        elif act == NotifyAct.FIN:
            fakecoopMap = np.array([1]).tobytes()
            coopMapType = 1
            bearCap = 1 
            await self.tx_handler.notify(AppConfig.id, did, AppConfig.topic, act, cid, fakecoopMap, coopMapType, bearCap) # TODO 这里不需要传协作图

    async def sendreq_send(self, did, cid, rl, pt, aoi, mode):
        await self.tx_handler.sendreq(did, cid, rl, pt, aoi, mode)

    async def send_send(self, sid, data):
        await self.tx_handler.send(sid, data)

    async def sendend_send(self, sid: AppConfig.sid_t):
        await self.tx_handler.sendend(sid)

    # ============== 处理收到的消息逻辑 ==============#
    async def broadcastpub_service(self, msg: BroadcastPubMessage):
        """
            收到广播推送
            1. 判断是否接受
            2. 如果接受，创建cctx，发送订阅请求
        """
        logging.debug(f"APP serve message {msg}")
        need = await self.check_need_broadcastpub(msg)
        if need:
            logging.info(f"接收 {msg.oid} 的BROADCASTPUB")
            cid = self.cid_gen()
            cctx = CContext(cid, msg.oid, AppConfig.id, None, self)
            self.add_cctx(cctx)
            await self.subscribe_send(msg.oid)
            cctx.to_waitnty()
            asyncio.create_task(self.cctx_loop(cctx))
        else:
            logging.info(f"拒绝 {msg.oid} 的BROADCASTPUB")

    async def broadcastsub_service(self, msg: BroadcastSubMessage):
        """
            收到广播订阅
            1. 判断是否接受
            2. 如果接受，发送广播订阅通知
        """
        logging.debug(f"APP serve message {msg}")
        need = await self.check_need_broadcastsub(msg)
        if need:
            logging.debug(f"接收 {msg.oid} 的BROADCASTSUB")
            coopMap = await self.get_my_conf_map()
            coopMapType = 1
            bearcap = 1
            # TODO: 这里有一个问题，如果还没有建立会话，就可能对同一个broadcastsub多次发送brocastsubnty
            # , 但broadcastsub发送时间间隔较长, 先忽略这个问题
            await self.tx_handler.brocastsubnty(AppConfig.id, msg.oid, AppConfig.topic, msg.context, 
                                            coopMap, coopMapType, bearcap)
        else:
            logging.debug(f"拒绝 {msg.oid} 的BROADCASTSUB")

    async def broadcastsubnty_service(self, msg: BroadcastSubNtyMessage):
        """
            收到广播订阅通知
            1. 寻找广播对话，因为收到了广播订阅，这个广播会话是自身创建的
            2. 检查是否需要对话
            3. 如果需要，创建cctx
        """
        logging.debug(f"APP serve message {msg}")
        bcctx = self.get_bcctx(msg.context)
        if bcctx == None:
            # 可能是超时或消息发送错误
            logging.warning(f"收到BROADCASTSUBNTY, 广播对话 context:{msg.context} 不存在")
            return
        server_assert(bcctx.state != BCContextState.PENDING)

        if bcctx.state == BCContextState.WAITBNTY:
            need = self.check_need(bcctx, msg)
            if need:
                logging.debug(f"接收 {msg.oid} 的BROADCASTSUBNTY")
                cctx = CContext(bcctx.cid, msg.oid, AppConfig.id, bcctx, self)
                self.add_cctx(cctx)
                bcctx.add_cctx(cctx)
                await self.subscribe_send(msg.oid, SubscribeAct.ACKUPD)
                cctx.to_waitnty()
            else:
                logging.debug(f"拒绝 {msg.oid} 的BROADCASTSUBNTY")
        elif bcctx.state == BCContextState.CLOSED:
            logging.warning(f"收到BROADCASTSUBNTY, 但对应 context:{msg.context} 已超时")
            pass

    async def notify_service(self, msg: NotifyMessage):
        """
            收到notify
            1. 寻找cctx，cctx是local发送订阅通知时创建的，local一定是cotee
            2. 根据消息内容，更新状态
        """
        logging.debug(f"APP serve message {msg}")
        cctx = self.get_cctx(msg.context, msg.oid, AppConfig.id)
        if cctx == None:
            server_logic_error(f"收到NOTIFY, 但对应 context:{msg.context} 不存在")
            return
        if cctx.is_cotor():
            server_logic_error(f"收到NOTIFY, 但对应 context:{msg.context} 中local是cotor")
            return
        server_assert(cctx.state != CContextCoteeState.PENDING)

        if cctx.state == CContextCoteeState.WAITNTY:
            if msg.act == NotifyAct.ACK:
                cctx.to_subscribing()
            elif msg.act == NotifyAct.FIN:
                cctx.to_closed()
            elif msg.act == NotifyAct.NTY:
                server_logic_error(f"收到NOTIFY.NTY, 但对应 context:{msg.context} 中状态是WAITNTY, 不应该收到NOTIFY.NTY")
        elif cctx.state == CContextCoteeState.SUBSCRIBING:
            if msg.act == NotifyAct.ACK:
                server_logic_error(f"收到NOTIFY.ACK, 但对应 context:{msg.context} 中状态是SUBSCRIBING, 不应该收到NOTIFY.ACK")
            elif msg.act == NotifyAct.FIN:
                cctx.to_closed()
            elif msg.act == NotifyAct.NTY:
                server_not_implemented(f"收到NOTIFY.ACK, 对应 context:{msg.context} 中状态是SUBSCRIBING")

    async def subscribe_ackupd_service(self, msg: SubscribeMessage):
        """
            收到订阅消息，act=subscribeAct=ACKUPD
            1. 此时自车不应该有cctx（目前只将ACKUPD看作ACK，并未实现UPD）
            2. 创建cctx，自车作为cotor
            3. 判断是否被订阅
            4. 发送notify
        """
        logging.debug(f"APP serve message {msg}")
        cctx = self.get_cctx(msg.context, AppConfig.id, msg.oid)
        if cctx is not None:
            logging.warning(f"收到SUBSCRIBE, 此时不应该存在此context:{msg.context}")
            return
        cctx = CContext(msg.context, AppConfig.id, msg.oid, None, self)
        cctx.to_sendnty()
        if await self.check_need_subscribe(msg):
            self.add_cctx(cctx)
            await self.notify_send(cctx.cid, msg.oid)
            cctx.to_subscribed()
        else:
            cctx.to_closed()

    async def subscribe_fin_service(self, msg: SubscribeMessage):
        """
            收到订阅消息，act=subscribeAct=FIN
            1. 此时自车应该有cctx
            2. 找到cctx
            3. 根据cctx状态行动
        """
        logging.debug(f"APP serve message {msg}")
        cctx = self.get_cctx(msg.context, AppConfig.id, msg.oid)
        if cctx is None:
            logging.warning(f"收到SUBSCRIBE FIN, 但不存在此context:{msg.context}")
            return
        if cctx.state == CContextCotorState.PENDING:
            # TODO 输出警告 代码错误 或消息错误
            logging.warning(f"收到SUBSCRIBE FIN, 但对应context:{msg.context}还未发送NOTIFY")
        elif cctx.state == CContextCotorState.SENDNTY:
            cctx.to_closed()
        elif cctx.state == CContextCotorState.SUBSCRIBED:
            cctx.to_closed()
        elif cctx.state == CContextCotorState.CLOSED:
            pass

    async def subscribe_service(self, msg: SubscribeMessage):
        logging.debug(f"APP serve message {msg}")
        if msg.act == SubscribeAct.ACKUPD:
            await self.subscribe_ackupd_service(msg)
        elif msg.act == SubscribeAct.FIN:
            await self.subscribe_fin_service(msg)
        else:
            assert False

    async def recvfile_service(self, msg: RecvFileMessage):
        logging.debug(f"APP serve message {msg}")
        cctx = self.get_cctx_or_panic(msg.context, AppConfig.id, msg.oid)
        assert cctx.state == CContextCoteeState.SUBSCRIBING
        data = read_binary_file(msg.file)
        self.data_cache[cctx.cid] = InfoDTO.InfoDTOSerializer.deserialize(data)

    async def sendfin_service(self, msg: SendFinMessage):
        logging.debug(f"APP serve message {msg}")

    async def sendrdy_service(self, msg: SendRdyMessage):
        logging.debug(f"APP serve message {msg}")
        cctx = self.get_cctx_or_panic(msg.context, AppConfig.id, msg.did)
        server_assert(not cctx.have_sid())
        cctx.sid = msg.sid
        cctx.stream_state = CSContextCotorState.SENDRDY
        self.add_stream(msg.sid, cctx)
        cctx.sid_set_event.set()
    
    async def recvrdy_service(self, msg: RecvRdyMessage):
        logging.debug(f"APP serve message {msg}")
        cctx = self.get_cctx_or_panic(msg.context, AppConfig.id, msg.oid)
        server_assert(not cctx.have_sid())
        cctx.sid = msg.sid
        cctx.stream_to_recvrdy()
        cctx.sid_set_event.set()

    async def recv_service(self, msg: RecvMessage):
        logging.debug(f"APP serve message {msg}")
        cctx = self.get_cctx_from_stream(msg.sid)
        if cctx is None:
            logging.warning(f'不存在与{msg.sid}关联的会话')
            return
        self.data_cache[cctx.cotee] = InfoDTO.InfoDTOSerializer.deserialize(msg.data)

    async def recvend_service(self, msg: RecvEndMessage):
        logging.debug(f"APP serve message {msg}")
        cctx = self.get_cctx_from_stream(msg.sid)
        if cctx is None:
            logging.warning(f'不存在与{msg.sid}关联的会话')
            return
        if cctx.stream_state == CSContextCoteeState.PENDING:
            server_logic_error(f"收到RECVEND, 会话context: {cctx.cid} 还未收到RECVRDY")
        elif cctx.stream_state == CSContextCoteeState.WAITRDY:
            server_logic_error(f"收到RECVEND, 会话context: {cctx.cid} 还未收到RECVRDY")
        elif cctx.stream_state == CSContextCoteeState.RECVRDY:
            cctx.stream_to_end()
        elif cctx.stream_state == CSContextCoteeState.RECVEND:
            logging.debug("收到RECVEND, 会话context: {cctx.cid} 流接收结束")

    # ============== 回调处理 ==============#
    async def appreg_recv(self, msg: AppRegMessage):
        logging.debug(f"APP Recv message {msg}")

    async def apprsp_recv(self, msg: AppRspMessage):
        logging.debug(f"APP Recv message {msg}")
        """处理注册响应 (MID.APPRSP)"""
        if result := msg.result:
            print(f"注册{'成功' if result else '失败'}")

    async def broadcastpub_recv(self, msg):
        logging.debug(f"APP Recv message {msg}")
        asyncio.create_task(self.broadcastpub_service(msg))

    async def broadcastsub_recv(self, msg: BroadcastSubMessage):
        logging.debug(f"APP Recv message {msg}")
        asyncio.create_task(self.broadcastsub_service(msg))

    async def broadcastsubnty_recv(self, msg: BroadcastSubNtyMessage):
        logging.debug(f"APP Recv message {msg}")
        bcctx = self.get_bcctx(msg.context)
        if bcctx is None:
            return
        await bcctx.msg_queue.put(msg)

    async def subscribe_recv(self, msg: SubscribeMessage):
        """
            收到subscribe消息
            1. ACKUPD
                新cctx
            2. FIN
                与现有的cctx关联
        """
        logging.debug(f"APP Recv message {msg}")
        if msg.act == SubscribeAct.ACKUPD:
            asyncio.create_task(self.subscribe_ackupd_service(msg))
        elif msg.act == SubscribeAct.FIN:
            await self.get_cctx_and_put(msg.context, AppConfig.id, msg.oid, msg)
        else:
            pass

    async def notify_recv(self, msg: NotifyMessage):
        logging.debug(f"APP Recv message {msg}")
        await self.get_cctx_and_put(msg.context, msg.oid, AppConfig.id, msg)

    async def sendreq_recv(self, msg: SendReqMessage):
        logging.debug(f"APP Recv message {msg}")

    async def sendrdy_recv(self, msg: SendRdyMessage):
        logging.debug(f"APP Recv message {msg}")
        await self.get_cctx_and_put(msg.context, AppConfig.id, msg.did, msg)

    async def recvrdy_recv(self, msg: RecvRdyMessage):
        logging.debug(f"APP Recv message {msg}")
        await self.get_cctx_and_put(msg.context, msg.oid, AppConfig.id, msg)

    async def send_recv(self, msg: SendMessage):
        logging.debug(f"APP Recv message {msg}")

    async def recv_recv(self, msg: RecvMessage):
        logging.debug(f"APP Recv message {msg}")
        asyncio.create_task(self.recv_recv(msg))

    async def sendend_recv(self, msg: SendEndMessage):
        logging.debug(f"APP Recv message {msg}")

    async def recvend_recv(self, msg: RecvEndMessage):
        logging.debug(f"APP Recv message {msg}")

    async def sendfile_recv(self, msg: SendFileMessage):
        logging.debug(f"APP Recv message {msg}")

    async def sendfin_recv(self, msg: SendFinMessage):
        logging.debug(f"APP Recv message {msg}")
        asyncio.create_task(self.sendfin_service(msg))

    async def recvfile_recv(self, msg: RecvFileMessage):
        logging.debug(f"APP Recv message {msg}")
        asyncio.create_task(self.recvfile_service(msg))
