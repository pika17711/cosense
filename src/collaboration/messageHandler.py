import asyncio
from dataclasses import dataclass
import json
import logging
import random
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from collaboration.message import NotifyAct, RecvEndMessage, RecvFileMessage, RecvMessage, RecvRdyMessage, SendFinMessage, SendRdyMessage, SubscribeAct
from collaboration.transactionHandler import transactionHandler
from utils import InfoDTO
import zmq
from zmq.asyncio import Context
from config import AppConfig

from enum import IntEnum, auto

from utils.common import load_json, mstime, read_binary_file, server_assert, string_to_32_hex, sync_to_async

from collaboration.messageID import MessageID
from cachetools import LRUCache

class CContextCotorState(IntEnum):
    """协作对话状态枚举"""
    PENDING = auto()         # 初始状态
    SENDNTY = auto()         # 收到订阅请求，未发出通知响应
    SUBSCRIBED = auto()      # 被订阅
    CLOSED = auto()          # 已终止

class CContextCoteeState(IntEnum):
    PENDING = auto()         # 初始状态
    WAITNTY = auto()         # 发出订阅请求，等待通知响应
    SUBSCRIBING = auto()     # 订阅中
    CLOSED = auto()          # 已终止

class BCContextState(IntEnum):
    PENDING = auto()         # 初始状态
    WAITBNTY = auto()        # 已发送广播订阅等待广播订阅通知
    CLOSED = auto()          # 已终止

class CSContextState(IntEnum):
    PENDING = auto()
    REQ = auto()
    RDY = auto()

"""
CollaborationContext
"""
class CContext:
    def __init__(self, cid: str, cotor, cotee, bcctx, mh):
        self.cid = cid                   # 对话ID
        self.cotor = cotor               # 协作者ID
        self.cotee = cotee               # 被协作者ID
        self.state = CContextCotorState.PENDING if self.is_cotor() else CContextCoteeState.PENDING
        self.bcctx: BCContext = bcctx    # 广播协作对话
        self.last_active = mstime()      # 最后活跃时间
        self.start_time = mstime()       # 开始时间
        self.queue = asyncio.Queue()
        self.message_handler: MessageHandler = mh
        self.sid_set_event = asyncio.Event()
        self.stream_state = CSContextState.PENDING
        self.sid = None


    def is_cotor(self):
        return self.cotor == AppConfig.id

    def is_cotee(self):
        return not self.is_cotor()

    def is_expired(self) -> bool:
        # 用上一次活跃的时间检查是否存活
        return (mstime() - self.last_active) > AppConfig.cctx_keepalive

    def remote_id(self):
        return self.cotee if self.is_cotor() else self.cotor

    def update_active(self):
        self.last_active = mstime()

    def is_alive(self) -> bool:
        return self.state not in (CContextCoteeState.CLOSED, CContextCotorState.CLOSED)

    def have_sid(self) -> bool:
        return self.sid is not None

    def to_sendnty(self):
        server_assert(self.is_cotor(), "上下文角色必须是协作者")
        server_assert(self.state == CContextCotorState.PENDING)
        self.update_active()
        self.state = CContextCotorState.SENDNTY

    def to_subscribed(self):
        server_assert(self.is_cotor(), "上下文角色必须是协作者")
        server_assert(self.state == CContextCotorState.SENDNTY)
        self.update_active()
        self.state = CContextCotorState.SUBSCRIBED
        self.message_handler.add_subscribed(self)

    def to_closed(self):
        self.update_active()
        self.state = CContextCoteeState.CLOSED if self.is_cotee() else CContextCotorState.CLOSED
        self.stream_close()
        self.message_handler.rem_subscribed(self)
        self.message_handler.rem_cctx(self)

    def force_close(self):
        """
            强行关闭
            无论当前状态是什么
        """
        self.state = CContextCoteeState.CLOSED if self.is_cotee() else CContextCotorState.CLOSED
        self.stream_close()
        self.message_handler.rem_subscribed(self)
        self.message_handler.rem_cctx(self)

    async def stream_get(self):
        # TODO 增加一个状态表示已经发出sendreq
        if self.stream_state == CSContextState.PENDING:
            rl = 1
            pt = 1
            aoi = 0
            mode = 1
            self.stream_state = CSContextState.REQ
            await self.message_handler.sendreq_send(self.remote_id(), self.cid, rl, pt, aoi, mode)
            await self.sid_set_event.wait()

    async def stream_close(self):
        if self.have_sid():
            await self.message_handler.sendend_send(self.sid)
            self.message_handler.rem_stream(self.sid)

    async def send_data(self, data):
        server_assert(self.is_cotee())
        if not self.have_sid():
            await self.stream_get()
        await self.message_handler.send_send(self.sid, data)

"""
BroadcastCollaborationContext
"""
class BCContext:
    def __init__(self, cid, mh):
        self.cid = cid
        self.set = set()
        self.start_time = mstime()
        self.state = BCContextState.PENDING
        self.queue = asyncio.Queue()
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
        assert self.state == BCContextState.PENDING, "必须发送了广播订阅后，才能等待广播通知消息"
        self.state = BCContextState.WAITBNTY
    
    def to_close(self):
        if self.state == BCContextState.PENDING:
            logging.warning("广播会话未发送广播订阅消息即被关闭")

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

        self.subscribing_set: Set[AppConfig.id_t] = set() # 正在订阅的cctx
        self.subscribed_set = Set[AppConfig.id_t] # 正在被订阅的cctx

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
            self.dispatch_message(msg)

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
        self.cctx_table.pop(cctx.cid, cctx.cotor, cctx.cotee)
        bcctx = cctx.bcctx
        if bcctx is not None:
            bcctx.rem_cctx(cctx)
    
    async def get_cctx_and_put(self, cid, cotor, cotee, msg: Message):
        cctx = self.get_cctx(cid, cotor, cotee)
        if cctx is None:
            return
        await cctx.queue.put(msg)

    def get_cctx_from_stream(self, sid) -> CContext:
        stream_table = self.stream_table
        if sid in stream_table:
            return stream_table[sid]
        else:
            return None

    async def add_stream(self, sid, cctx):
        self.stream_table[sid] = cctx

    async def rem_stream(self, sid):
        self.stream_table.pop(sid)

    # ============== bcctx ====================#
    def add_bcctx(self, bcctx: BCContext):
        self.bcctx_table[bcctx.cid] = bcctx

    def get_bcctx(self, cid: AppConfig.cid_t) -> BCContext:
        ctable = self.bcctx_table
        try:
            cctx_set = ctable[cid]
        except KeyError:
            cctx_set = None

        return cctx_set

    def rem_bcctx(self, bcctx: BCContext):
        self.bcctx_table.pop(bcctx.cid)

    # ============== sub ====================#
    def add_subscribing(self, cctx: CContext):
        self.subscribing_set.add(cctx)

    def rem_subscribing(self, cctx: CContext):
        self.subscribing_set.remove(cctx)

    def get_subscribing(self):
        return self.subscribing_set

    def add_subscribed(self, cctx: CContext):
        self.subscribed_set.add(cctx)

    def rem_subscribed(self, cctx: CContext):
        self.subscribed_set.remove(cctx)

    def get_subscribed(self) -> Set[CContext]:
        return self.subscribed_set

    # ============== utils ====================#
    async def wait_with_timeout(self, coro, timeout: int):
        try:
            return await asyncio.wait_for(coro, timeout=timeout/1000)
        except asyncio.TimeoutError:
            return None

    def check_need_pub(self, msg: BroadcastPubMessage):
        """是否需要别人的广播推送"""
        # TODO 逻辑
        return True

    def check_need(self, bcctx: BCContext, msg: Message):
        # 检查是否需要的逻辑
        # TODO
        return True

    # 判断是否能够被订阅
    async def check_subscribed(self):
        # TODO
        return True
    
    def cid_gen(self):
        self.cid_counter += 1
        return string_to_32_hex(str(AppConfig.app_id) + str(mstime()) + str(self.cid_counter))

    # ============== ctx loop ====================#
    async def ctx_loop(self, ctx: Union[CContext, BCContext], 
                       handler_table: Dict[MessageID, Callable[[Message], Coroutine[Any, Any, None]]],
                       rem_ctx: Callable[[Union[CContext, BCContext]], None]):
        while ctx.is_alive():
            msg = await self.wait_with_timeout(ctx.queue.get(), 100) # 随便取的100ms，只会影响超时的准确性
            if msg is not None and msg.header.mid in handler_table:
                await handler_table[msg.header.mid](msg)
            if ctx.is_expired():
                ctx.force_close()

    async def bcctx_loop(self, bcctx: BCContext):
        handler_table = {BroadcastSubMessage: self.broadcastsub_service,
                         BroadcastSubNtyMessage: self.broadcastsubnty_service}
        await self.ctx_loop(bcctx, handler_table, self.rem_bcctx)

    async def cctx_loop(self, cctx: CContext):
        handler_table = {SubscribeMessage: self.subscribe_service,
                         NotifyMessage: self.notify_service}
        await self.ctx_loop(cctx, handler_table, self.rem_cctx)

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
        bcctx.state = BCContextState.PENDING
        self.add_bcctx(bcctx)
        await self.tx_handler.brocastsub(AppConfig.id, AppConfig.topic, cid, coopMap, coopMapType, bearCap)
        bcctx.state = BCContextState.WAITBNTY
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
                cctx = CContext(cid, AppConfig.id, didi, None)
                cctx.state = CContextCoteeState.PENDING
                self.add_cctx(cctx)
                coopMap = await self.get_my_conf_map()
                coopMapType = 1
                bearCap = 1
                await self.tx_handler.subscribe(AppConfig.id, didi, AppConfig.topic, act, cid, coopMap, coopMapType, bearCap)
                cctx.state = CContextCoteeState.WAITNTY
                await asyncio.create_task(self.cctx_loop(cctx.cid, cctx.cotor, cctx.cotee, self.notify_service))
            else:
                cctx = self.get_cctx(cid, AppConfig.id, didi)
                await self.tx_handler.subscribe(AppConfig.id, didi, AppConfig.topic, act, cid, None, None, 0)
                cctx.force_close()
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
        cctx = self.get_cctx_or_panic(cid, did, AppConfig.id)
        if act == NotifyAct.ACK:
            coopMap = await self.get_my_conf_map()
            coopMapType = 1
            bearCap = 1
            await self.tx_handler.notify(AppConfig.id, did, AppConfig.topic, act, cid, coopMap, coopMapType, bearCap)
            cctx.state = CContextCotorState.SUBSCRIBED
        elif act == NotifyAct.NTY:
            pass
        elif act == NotifyAct.FIN:
            cctx.state = CContextCotorState.CLOSED
            self.rem_cctx(cctx)

    async def sendreq_send(self, did, cid, rl, pt, aoi, mode):
        await self.tx_handler.sendreq(did, cid, rl, pt, aoi, mode)

    async def send_send(self, sid, data):
        await self.tx_handler.send(sid, data)

    async def sendend_send(self, sid):
        await self.tx_handler.sendend(sid)

    # ============== 处理收到的消息逻辑 ==============#
    async def broadcastpub_service(self, msg: BroadcastPubMessage):
        """
            收到广播推送
            1. 判断是否接受
            2. 如果接受，创建cctx，发送订阅请求
        """
        need = await self.check_need_pub(msg)
        if need:
            cid = self.cid_gen()
            cctx = CContext(cid, msg.oid, AppConfig.id, None)
            self.add_cctx(cctx)
            await self.subscribe_send(msg.oid)
            cctx.state = CContextCoteeState.WAITNTY
            asyncio.create_task(self.cctx_loop(cctx))

    async def broadcastsub_service(self, msg: BroadcastSubMessage):
        """
            收到广播订阅
            1. 判断是否接受
            2. 如果接受，创建cctx，发送广播订阅通知
        """
        subed = await self.check_subscribed()
        if not subed:
            return
        coopMap = await self.get_my_conf_map()
        coopMapType = 1
        bearcap = 1
        self.tx_handler.brocastsubnty(AppConfig.id, msg.oid, AppConfig.topic, msg.context, 
                                        coopMap, coopMapType, bearcap)

    async def broadcastsubnty_service(self, msg: BroadcastSubNtyMessage):
        """
            收到广播订阅通知
            1. 寻找广播对话，因为收到了广播订阅，这个广播会话是自身创建的
            2. 检查是否需要对话
            3. 如果需要，创建cctx
        """
        bcctx = self.get_bcctx(msg.context)
        if bcctx == None:
            # 有可能是广播对话超时，有可能是消息发送错误
            # TODO 输出警告
            return

        assert bcctx.state != BCContextState.PENDING
        if bcctx.state == BCContextState.WAITBNTY:
            need = self.check_need(bcctx, msg)
            if need:
                cctx = CContext(bcctx.cid, msg.oid, AppConfig.id, bcctx)
                self.add_cctx(cctx)
                bcctx.add_cctx(cctx)
                self.subscribe_send(AppConfig.id, msg.oid, SubscribeAct.ACKUPD)
                cctx.state = CContextCoteeState.WAITNTY
            else:
                # 不需要
                # TODO 输出警告
                pass
        elif bcctx.state == BCContextState.CLOSED:
            # 已经关闭，不再接收连接
            # TODO 输出警告
            pass

    async def notify_service(self, msg: NotifyMessage):
        """
            收到notify
            1. 寻找cctx，cctx是自车发送订阅通知时创建的，自车一定是cotee
            2. 根据消息内容，更新状态
        """
        cctx = self.get_cctx(msg.context, msg.oid, AppConfig.id)
        if cctx == None:
            # 输出警告
            return
        if cctx.is_cotor():
            # 消息错误或者代码逻辑错误
            return
        assert cctx.state != CContextCoteeState.PENDING

        cctx.last_active = mstime()
        if cctx.state == CContextCoteeState.WAITNTY:
            if msg.act == NotifyAct.ACK:
                cctx.state == CContextCoteeState.SUBSCRIBING
            elif msg.act == NotifyAct.FIN:
                cctx.state == CContextCoteeState.CLOSED
                self.rem_cctx(cctx)
            elif msg.act == NotifyAct.NTY:
                # TODO 输出警告：当前状态是WAITNTY，不应该收到notify:nty
                pass
        elif cctx.state == CContextCoteeState.SUBSCRIBING:
            if msg.act == NotifyAct.ACK:
                # TODO 输出警告：当前状态是SUBSCRIBING，不应该收到notify:ACK
                pass
            elif msg.act == NotifyAct.FIN:
                cctx.state == CContextCoteeState.CLOSED
                self.rem_cctx(cctx)
            elif msg.act == NotifyAct.NTY:
                # 收到这个消息是合理的，未实现
                pass

    async def subscribe_ackupd_service(self, msg: SubscribeMessage):
        """
            收到订阅消息，act=subscribeAct=ACKUPD
            1. 此时自车不应该有cctx
            2. 创建cctx，自车作为cotor
            3. 判断是否被订阅
            4. 发送notify
        """
        cctx = self.get_cctx(msg.context, AppConfig.id, msg.oid)
        if cctx is not None:
            # 输出警告
            return
        cctx = CContext(msg.context, AppConfig.id, msg.oid, None)
        cctx.state = CContextCotorState.SENDNTY
        if await self.check_subscribed():
            self.add_cctx(cctx)
            await self.notify_send(cctx.cid, msg.oid)
            cctx.state = CContextCotorState.SUBSCRIBED

    async def subscribe_fin_service(self, msg: SubscribeMessage):
        """
            收到订阅消息，act=subscribeAct=FIN
            1. 此时自车应该有cctx
            2. 找到cctx
            3. 根据cctx状态行动
        """
        cctx = self.get_cctx(msg.context, AppConfig.id, msg.oid)
        if cctx is None:
            # 输出警告
            return
        if cctx.state == CContextCotorState.PENDING:
            # TODO 输出警告 代码错误 或消息错误
            pass
        elif cctx.state == CContextCotorState.SENDNTY:
            cctx.state = CContextCotorState.CLOSED
            self.rem_cctx(cctx)
        elif cctx.state == CContextCotorState.SUBSCRIBED:
            cctx.state = CContextCotorState.CLOSED
            self.rem_cctx(cctx)
        elif cctx.state == CContextCotorState.CLOSED:
            pass

    async def subscribe_service(self, msg: SubscribeMessage):
        if msg.act == SubscribeAct.ACKUPD:
            self.subscribe_ackupd_service(msg)
        elif msg.act == SubscribeAct.FIN:
            self.subscribe_fin_service(msg)
        else:
            assert False

    async def recvfile_service(self, msg: RecvFileMessage):
        cctx = self.get_cctx_or_panic(msg.context, AppConfig.id, msg.oid)
        assert cctx.state == CContextCoteeState.SUBSCRIBING
        data = read_binary_file(msg.file)
        self.data_cache[cctx.cid] = InfoDTO.InfoDTOSerializer.deserialize(data)

    async def sendfin_service(self, msg: SendFinMessage):
        pass

    async def sendrdy_service(self, msg: SendRdyMessage):
        cctx = self.get_cctx_or_panic(msg.context, AppConfig.id, msg.did)
        server_assert(not cctx.have_sid())
        cctx.sid = msg.sid
        cctx.stream_state = CSContextState.RDY
        cctx.sid_set_event.set()
    
    async def recvrdy_service(self, msg: RecvRdyMessage):
        cctx = self.get_cctx_or_panic(msg.context, AppConfig.id, msg.oid)
        server_assert(not cctx.have_sid())
        cctx.sid = msg.sid
        cctx.sid_set_event.set()

    async def recv_service(self, msg: RecvMessage):
        cctx = self.get_cctx_from_stream(msg.sid)
        if cctx is None:
            logging.warning(f'不存在与{msg.sid}关联的会话')
            return
        self.data_cache[cctx.cotee] = InfoDTO.InfoDTOSerializer.deserialize(msg.data)

    async def recvend_service(self, msg: RecvEndMessage):
        cctx = self.get_cctx_from_stream(msg.sid)
        if cctx is None:
            logging.warning(f'不存在与{msg.sid}关联的会话')
            return

    # ============== 回调处理 ==============#
    async def appreg_recv(self, msg):
        print(f"Received APPREG message: {msg}")

    async def apprsp_recv(self, msg):
        """处理注册响应 (MID.APPRSP)"""
        if result := msg.get('result'):
            print(f"注册{'成功' if result else '失败'}")

    async def broadcastpub_recv(self, msg):
        logging.info(f"Received BROCASTPUB message: {msg}")
        await asyncio.create_task(self.broadcastpub_service(msg))

    async def broadcastsub_recv(self, msg: BroadcastSubMessage):
        logging.info(f"Received BROCASTSUBNTY message: {msg}")
        await asyncio.create_task(self.broadcastsub_service(msg))

    async def broadcastsubnty_recv(self, msg: BroadcastSubNtyMessage):
        logging.info(f"Received BROCASTSUB message: {msg}")
        bcctx = self.get_bcctx(msg.context)
        if bcctx is None:
            return
        await bcctx.queue.put(msg)

    async def subscribe_recv(self, msg: SubscribeMessage):
        """
            收到subscribe消息
            1. ACKUPD
                新cctx
            2. FIN
                与现有的cctx关联
        """
        logging.info(f"Received SUBSCRIBE message: {msg}")
        if msg.act == SubscribeAct.ACKUPD:
            await asyncio.create_task(self.subscribe_ackupd_service(msg))
        elif msg.act == SubscribeAct.FIN:
            self.get_cctx_and_put(msg.context, msg.oid, AppConfig.id)
        else:
            pass

    async def notify_recv(self, msg: NotifyMessage):
        logging.info(f"Received NOTIFY message: {msg}")
        self.get_cctx_and_put(msg.context, msg.oid, AppConfig.id)

    async def sendreq_recv(self, msg):
        logging.info(f"Received SENDREQ message: {msg}")

    async def sendrdy_recv(self, msg):
        logging.info(f"Received SENDRDY message: {msg}")
        await asyncio.create_task(self.sendrdy_service(msg))

    async def recvrdy_recv(self, msg):
        logging.info(f"Received RECVRDY message: {msg}")
        await asyncio.create_task(self.recvrdy_service(msg))

    async def send_recv(self, msg):
        logging.info(f"Received SEND message: {msg}")

    async def recv_recv(self, msg):
        logging.info(f"Received RECV message: {msg}")
        await asyncio.create_task(self.recv_recv(msg))

    async def sendend_recv(self, msg):
        logging.info(f"Received SENDEND message: {msg}")

    async def recvend_recv(self, msg):
        logging.info(f"Received RECVEND message: {msg}")

    async def sendfile_recv(self, msg):
        logging.info(f"Received SENDFILE message: {msg}")

    async def sendfin_recv(self, msg):
        logging.info(f"Received SENDFIN message: {msg}")
        await asyncio.create_task(self.sendfin_service(msg))

    async def recvfile_recv(self, msg):
        """处理文件接收 (MID.RECVFILE)"""
        await asyncio.create_task(self.recvfile_service(msg))
