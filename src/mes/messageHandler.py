import asyncio
import json
import logging
import random
from typing import Dict, List, Optional, Tuple, Union
from src.mes.transactionHandler import notifyAct, subscribeAct, transactionHandler
import zmq
from zmq.asyncio import Context
from src.config import CONFIG

from enum import IntEnum, auto

from utils.asyncVariable import AsyncVariable
from utils.common import mstime

from src.mes.mid import MessageID

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

class CContextSetState(IntEnum):
    PENDING = auto()            # 初始状态
    WAITBNTY = auto()        # 已发送广播订阅等待广播订阅通知
    CLOSED = auto()          # 已终止

"""
CollaborationContext
"""
class CContext:
    def __init__(self, cid: str, cotor, cotee, cctxset):
        self.cid = cid               # 对话ID
        self.cotor = cotor           # 协作者ID
        self.cotee = cotee           # 被协作者ID
        self.state = None
        self.cctxset: CContextSet = cctxset       # 协作对话集
        self.last_active = mstime()  # 最后活跃时间
        self.start_time = mstime()   # 开始时间
        self.queue = asyncio.Queue()

    def is_cotor(self):
        return self.cotor == AppConfig.id

    def is_cotee(self):
        return not self.is_cotor()

    def is_expired(self) -> bool:
        return (mstime() - self.last_active) > AppConfig.cctx_keepalive

    def is_alive(self) -> bool:
        return self.state in (CContextCoteeState.CLOSED, CContextCotorState.CLOSED)

"""
BroadcastCollaborationContext
"""
class CContextSet:
    def __init__(self, cid):
        self.cid = cid
        self.set = set()
        self.start_time = mstime()
        self.state = CContextSetState.PENDING
        self.queue = asyncio.Queue()

    def add_cctx(self, cctx: CContext):
        self.set.add(cctx)
    
    def rem_cctx(self, cctx: CContext):
        self.set.remove(cctx)

    def is_expired(self) -> bool:
        return mstime() - self.start_time > AppConfig.cctxset_keepalive

    def is_alive(self) -> bool:
        return self.state == CContextSetState.CLOSED

"""
CollaborationStreamContext
"""
class CSContext:
    def __init__(self, sid, cctx):
        self.sid = sid
        self.cctx = cctx
        self.state = CContextSetState.PENDING
        self.queue = asyncio.Queue()

from src.mes.message import BroadcastPubMessage, Message, NotifyMessage, SubscribeMessage
from src.mes.message import BroadcastSubNtyMessage, BroadcastSubMessage
from config import AppConfig

class MessageHandler:
    def __init__(self):
        self.msg_queue = asyncio.Queue()
        self.tx_handler: transactionHandler = transactionHandler(self.msg_queue)

        self.cctx_table: Dict[Tuple[AppConfig.cid_t, AppConfig.id_t, AppConfig.id_t], CContext] = dict() # (cid, cotor, cotee) -> cctx
        self.cctxset_table: Dict[AppConfig.cid_t, CContextSet] = dict() # cid -> cctxset
        self.sctx_table: Dict[AppConfig.sid_t, CSContext] = dict() # sid -> csctx

        self.subscribing_set = set() # 正在订阅的cctx
        self.subscribed_set = set() # 正在被订阅的cctx

        self.cid_counter = random.randint(1, 100000)


    async def recv_loop(self):
        await self.tx_handler.recv_loop()

    async def dispatch(self):
        """消息路由分发"""
        while True:
            msg = await self.msg_queue.get()
            self._handle_message(msg)

    async def _handle_message(self, msg: Message):
        mid = msg.header.mid
        if mid == MessageID.APPREG:
            await self.recv_appreg(msg)
        elif mid == MessageID.APPRSP:
            await self.recv_apprsp(msg)
        elif mid == MessageID.BROCASTPUB:
            await self.recv_broadcastpub(msg)
        elif mid == MessageID.BROCASTSUB:
            await self.recv_broadcastsub(msg)
        elif mid == MessageID.BROCASTSUBNTY:
            await self.recv_broadcastsubnty(msg)
        elif mid == MessageID.SUBSCRIBE:
            await self.recv_subscribe(msg)
        elif mid == MessageID.NOTIFY:
            await self.recv_notify(msg)
        elif mid == MessageID.SENDREQ:
            await self.recv_sendreq(msg)
        elif mid == MessageID.SENDRDY:
            await self.recv_sendrdy(msg)
        elif mid == MessageID.RECVRDY:
            await self.recv_recvrdy(msg)
        elif mid == MessageID.SEND:
            await self.recv_send(msg)
        elif mid == MessageID.RECV:
            await self.recv_recv(msg)
        elif mid == MessageID.SENDEND:
            await self.recv_sendend(msg)
        elif mid == MessageID.RECVEND:
            await self.recv_recvend(msg)
        elif mid == MessageID.SENDFILE:
            await self.recv_sendfile(msg)
        elif mid == MessageID.SENDFIN:
            await self.recv_sendfin(msg)
        elif mid == MessageID.RECVFILE:
            await self.recv_recvfile(msg)
        else:
            logging.warning(f"Unhandled message type: {MessageID.get_name(mid.value)}")

    # ============== cctx ====================#
    def add_cctx(self, cctx: CContext):
        self.cctx_table[(cctx.cid, cctx.cotor, cctx.cotee)] = cctx

    def check_cctx_exist(self, cid, cotor, cotee):
        return self.get_cctx(cid, cotor, cotee) is not None

    def get_cctx(self, cid, cotor, cotee) -> CContext:
        ctable = self.cctx_table
        try:
            cctx = ctable[(cid, cotor, cotee)]
        except KeyError:
            cctx = None

        return cctx

    def rem_cctx(self, cctx: CContext):
        self.cctx_table.pop(cctx.cid, cctx.cotor, cctx.cotee)
        cctxset = cctx.cctxset
        if cctxset is not None:
            cctxset.rem_cctx(cctx)
        # 断开连接
    
    async def get_cctx_and_put(self, cid, cotor, cotee, msg: Message):
        cctx = self.get_cctx(cid, cotor, cotee)
        if cctx is None:
            return
        await cctx.queue.put(msg)

    # ============== cctxset ====================#
    def add_cctxset(self, cctxset: CContextSet):
        self.cctxset_table[cctxset.cid] = cctxset

    def get_cctxset(self, cid: AppConfig.cid_t) -> CContextSet:
        ctable = self.cctxset_table
        try:
            cctx_set = ctable[cid]
        except KeyError:
            cctx_set = None

        return cctx_set

    def rem_cctxset(self, cctxset: CContextSet):
        self.cctxset_table.pop(cctxset.cid)

    # ============== sub ====================#
    def add_subscribing(self, cctx: CContext):
        self.subscribing_set.add(cctx)

    def rem_subscribing(self, cctx: CContext):
        self.subscribing_set.add(cctx)

    def add_subscribed(self, cctx: CContext):
        self.subscribed_set.add(cctx)

    def rem_subscribed(self, cctx: CContext):
        self.subscribed_set.add(cctx)

    # ============== utils ====================#
    async def wait_with_timeout(self, coro, timeout: int):
        try:
            return await asyncio.wait_for(coro, timeout=timeout/1000)
        except asyncio.TimeoutError:
            return None

    def check_need_pub(self, msg):
        # TODO 逻辑
        return True
    
    
    def check_need(self, cctxset: CContextSet, msg: Message):
        # 检查是否需要的逻辑
        return True

    # 判断是否能够被订阅
    async def check_subscribed(self):
        # TODO
        return True
    
    def cid_gen(self):
        self.cid_counter += 1
        return str(AppConfig.app_id) + str(mstime()) + str(self.cid_counter)


    # ============== custom loop ====================#
    async def cctxset_loop(self, cctxset: CContextSet):
        handler_table = {BroadcastSubMessage: self.broadcastsub_service,
                         BroadcastSubNtyMessage: self.broadcastsubnty_service}
        while cctxset.is_alive():
            msg = await self.wait_with_timeout(cctxset.queue.get(), 100) # 随便取的100ms
            msg: Union[BroadcastSubMessage, BroadcastSubNtyMessage]
            try:
                await handler_table[msg.header.mid](msg)
            except KeyError:
                assert False

            if cctxset.is_expired():
                self.rem_cctxset(cctxset)

    async def cctx_loop(self, cctx: CContext):
        handler_table = {SubscribeMessage: self.subscribe_service,
                         NotifyMessage: self.notify_service}
        while cctx.is_alive():
            msg = await self.wait_with_timeout(cctx.queue.get(), 100) # 随便取的100ms
            msg: Union[SubscribeMessage, NotifyMessage]
            if msg.header.mid in handler_table:
                handler_table[msg.header.mid](msg)
            else:
                assert False
            if cctx.is_expired():
                self.rem_cctx(cctx)

    # ============== 发消息逻辑 ====================#
    async def broadcastsub_send(self):
        cid = self.cid_gen()
        coopMap = None
        coopMapType = None
        bearCap = 1
        cctxset = CContextSet(cid)
        cctxset.state = CContextSetState.PENDING
        self.add_cctxset(cctxset)
        await self.tx_handler.brocastsub(AppConfig.id, AppConfig.topic, cid, coopMap, coopMapType, bearCap)
        cctxset.state = CContextSetState.WAITBNTY
        asyncio.create_task(self.cctxset_loop(cctxset))

    async def broadcastpub_send(self):
        coopMap = None
        coopMapType = None
        await self.tx_handler.brocastpub(AppConfig.id, AppConfig.topic, coopMap, coopMapType)

    async def subscribe_send(self, did: Union[List[AppConfig.id_t], AppConfig.id_t], act=subscribeAct.ACKUPD):
        if type(did) is AppConfig.id_t:
            did = [did]
        for didi in did:
            cid = self.cid_gen()
            cctx = CContext(cid, AppConfig.id, didi, None)
            cctx.state = CContextCoteeState.PENDING
            self.add_cctx(cctx)
            coopMap = None
            coopMapType = None
            bearCap = 1
            await self.tx_handler.subscribe(AppConfig.id, didi, AppConfig.topic, act, cid, coopMap, coopMapType, bearCap)
            cctx.state = CContextCoteeState.WAITNTY
            await asyncio.create_task(self.cctx_loop(cctx.cid, cctx.cotor, cctx.cotee, self.notify_service))

    async def notify_send(self, cid, did, act=notifyAct.ACK):
        cctx = self.get_cctx(cid, did, AppConfig.id)
        coopMap = None
        coopMapType = None
        bearCap = 1
        await self.tx_handler.notify(AppConfig.id, did, AppConfig.topic, act, cid, coopMap, coopMapType, bearCap)
        if act == notifyAct.ACK:
            cctx.state = CContextCotorState.SUBSCRIBED
        elif act == notifyAct.NTY:
            pass
        elif act == notifyAct.FIN:
            cctx.state = CContextCotorState.CLOSED

    # ============== 处理收到的消息逻辑 ==============#
    async def broadcastpub_service(self, msg: BroadcastPubMessage):
        need = await self.check_need_pub(msg)
        if need:
            cid = self.cid_gen()
            cctx = CContext(cid, msg.oid, AppConfig.id, None)
            cctx.state = CContextCoteeState.PENDING
            self.add_cctx(cctx)
            await self.subscribe_send(msg.oid)
            cctx.state = CContextCoteeState.WAITNTY
            asyncio.create_task(self.cctx_loop(cid, msg.oid, AppConfig.id, self.notify_service()))

    async def broadcastsub_service(self, msg: BroadcastSubMessage):
        subed = await self.check_subscribed()
        if subed:
            coodMap = None # TODO
            coodMapType = None # TODO
            bearcap = 1
            status = self.tx_handler.brocastsubnty(AppConfig.id, msg.oid, AppConfig.topic, msg.context, 
                                          coodMap, coodMapType, bearcap)
            if not status:
                # 事务失败了，说明从应用到通信模块发送失败了 
                # TODO 重试机制
                pass

    async def broadcastsubnty_service(self, msg: BroadcastSubNtyMessage):
        cctxset = self.get_cctxset(msg.context)
        if cctxset == None:
            # 输出警告
            return

        if cctxset.state == CContextSetState.WAITBNTY:
            need = self.check_need(cctxset, msg)
            if need:
                cctx = CContext(cctxset.cid, msg.oid, AppConfig.id, cctxset)
                cctx.state = CContextCotorState.SENDNTY
                cctxset.add_cctx(cctx)
            else:
                # 输出警告
                pass
        elif cctxset.state == CContextSetState.PENDING:
            assert False
            # 加入错误信息
        elif cctxset.state == CContextSetState.CLOSED:
            # 已经关闭，不再接收连接
            pass

    async def notify_service(self, msg: NotifyMessage):
        cctx = self.get_cctx(msg.context, msg.oid, AppConfig.id)
        if cctx == None:
            # 输出警告
            return
        if cctx.is_cotee():
            if cctx.state == CContextCoteeState.WAITNTY:
                cctx.state == CContextCoteeState.SUBSCRIBING
                cctx.last_active = mstime()
            else:
                # 自车是被协助者，收到notify时是其他状态是不可能的，消息不合理
                pass
        else:
            # 自车是协助者，不可能收到notify
            pass

    async def subscribe_ackupd_service(self, msg: SubscribeMessage):
        cctx = self.get_cctx(msg.context, AppConfig.id, msg.oid)
        if cctx is not None:
            # 输出警告
            return
        cctx = CContext(msg.context, AppConfig.id, msg.oid, None)
        cctx.state = CContextCotorState.SENDNTY
        if await self.check_subscribed():
            self.add_cctx(cctx)
            coopMap = None # TODO
            coopMapType = None # TODO
            bearcap = 1
            await self.tx_handler.notify(AppConfig.id, msg.oid, AppConfig.topic, notifyAct.ACK, msg.context, coopMap, coopMapType, bearcap)
            cctx.state = CContextCotorState.SUBSCRIBED


    async def subscribe_fin_service(self, msg: SubscribeMessage):
        cctx = self.get_cctx(msg.context, AppConfig.id, msg.oid)
        if cctx is None:
            # 输出警告
            return
        if cctx.state == CContextCotorState.SUBSCRIBED:
            cctx.state = CContextCotorState.CLOSED
            self.rem_cctx(cctx)

    async def subscribe_service(self, msg: SubscribeMessage):
        if msg.action == subscribeAct.ACKUPD:
            self.subscribe_ackupd_service(msg)
        elif msg.action == subscribeAct.FIN:
            self.subscribe_fin_service(msg)
        else:
            assert False

    # ============== 回调处理 ==============#
    async def recv_appreg(self, msg):
        print(f"Received APPREG message: {msg}")

    async def recv_apprsp(self, msg):
        """处理注册响应 (MID.APPRSP)"""
        if result := msg.get('result'):
            print(f"注册{'成功' if result else '失败'}")

    async def recv_broadcastpub(self, msg):
        logging.info(f"Received BROCASTPUB message: {msg}")
        await asyncio.create_task(self.broadcastpub_service(msg))

    async def recv_broadcastsub(self, msg: BroadcastSubMessage):
        logging.info(f"Received BROCASTSUBNTY message: {msg}")
        await asyncio.create_task(self.broadcastsub_service(msg))

    async def recv_broadcastsubnty(self, msg: BroadcastSubNtyMessage):
        logging.info(f"Received BROCASTSUB message: {msg}")
        cctxset = self.get_cctxset(msg.context)
        if cctxset is None:
            return
        await cctxset.queue.put(msg)

    async def recv_subscribe(self, msg: SubscribeMessage):
        """
            收到subscribe消息
            1. ACKUPD
                新cctx
            2. FIN
                与现有的cctx关联
        """
        logging.info(f"Received SUBSCRIBE message: {msg}")
        if msg.action == subscribeAct.ACKUPD:
            await asyncio.create_task(self.subscribe_ackupd_service(msg))
        elif msg.action == subscribeAct.FIN:
            self.get_cctx_and_put(msg.context, msg.oid, AppConfig.id)
        else:
            pass

    async def recv_notify(self, msg: NotifyMessage):
        logging.info(f"Received NOTIFY message: {msg}")
        self.get_cctx_and_put(msg.context, msg.oid, AppConfig.id)

    async def recv_sendreq(self, msg):
        logging.info(f"Received SENDREQ message: {msg}")

    async def recv_sendrdy(self, msg):
        logging.info(f"Received SENDRDY message: {msg}")

    async def recv_recvrdy(self, msg):
        logging.info(f"Received RECVRDY message: {msg}")

    async def recv_send(self, msg):
        logging.info(f"Received SEND message: {msg}")

    async def recv_recv(self, msg):
        """处理流数据接收 (MID.RECV)"""
        sid = msg.get('sid')
        data = bytes.fromhex(msg.get('data', ''))

        if sid not in self.stream_ctx:
            self.stream_ctx[sid] = {'buffer': b''}

        self.stream_ctx[sid]['buffer'] += data
        if len(data) < CONFIG.get('stream_chunk_size', 1024):
            await self._process_stream(sid)

    async def recv_sendend(self, msg):
        logging.info(f"Received SENDEND message: {msg}")

    async def recv_recvend(self, msg):
        logging.info(f"Received RECVEND message: {msg}")

    async def recv_sendfile(self, msg):
        logging.info(f"Received SENDFILE message: {msg}")

    async def recv_sendfin(self, msg):
        logging.info(f"Received SENDFIN message: {msg}")

    async def recv_recvfile(self, msg):
        """处理文件接收 (MID.RECVFILE)"""
        file_path = msg.get('file')
        logging.info(f"收到文件保存至: {file_path}")

    async def _process_stream(self, sid: int):
        """处理完整流数据"""
        data = self.stream_ctx[sid]['buffer']
        logging.info(f"处理流数据 {sid} 长度: {len(data)}")
        del self.stream_ctx[sid]