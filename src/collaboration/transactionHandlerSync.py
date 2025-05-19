from __future__ import annotations

import logging
import queue
import random
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, Thread
from typing import Callable, Dict, List, Optional
from queue import Queue

# 假设这些类和模块在其他地方定义
import appType
from collaboration.messageID import MessageID
from collaboration.message import AckMessage, Message
from collaboration.ICP import ICPServer, ICPClient
from appConfig import AppConfig
from utils.common import ms2s

class txContext:
    def __init__(self, tid):
        self.tid = tid
        self.event = Event()
        self.response: Optional[AckMessage] = None

class transactionHandlerSync:
    """
        事务处理器
            1. 实现tid相关部分, 保证消息在应用向通信子系统发送的消息可靠传输.
            2. 进行消息接收和解析, 
            TODO: 
                1. 保证消息在通信子系统向应用发送的消息可靠传输.
                2. 失败重试
    """
    def __init__(self, 
                 cfg: AppConfig, 
                 icp_server: ICPServer,
                 icp_client: ICPClient,
                 ):
        self.cfg = cfg
        self.recv_queue = Queue()
        self.tid_counter = random.randint(0, 1 << 20)
        self.tx_table: Dict[int, txContext] = dict()  # tid -> txContext
        self.icp_server = icp_server
        self.icp_client = icp_client
        self.running = True
        self.msg_queue = Queue()
        self.lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=100)
        self.recv_thread = Thread(target=self.recv_loop, name='transactionHandler recv_loop', daemon=True)

    def start_recv(self):
        self.recv_thread.start()

    def close(self):
        self.running = False
        self.executor.shutdown()
        if self.recv_thread.is_alive():
            self.recv_thread.join(timeout=1.0)

    def new_tid(self):
        with self.lock:
            self.tid_counter += 1
            return self.tid_counter

    def add_tx(self, tid, txctx):
        with self.lock:
            self.tx_table[tid] = txctx

    def rem_tx(self, tid):
        with self.lock:
            if tid in self.tx_table:
                return self.tx_table.pop(tid)
            else:
                return None

    def recv_loop(self):
        while self.running:
            try:
                resp = self.icp_client.recv_message()
                if resp is None:
                    continue
                mes = Message.parse(resp)
                logging.debug(f"recv message: {mes}")
                if mes.header.mid == MessageID.ACK:
                    self.ack_resp_handler(mes)
                else:
                    if MessageID.is_control(mes.header.mid):
                        # TODO 发送ACK消息给通信模块
                        pass
                    self.msg_queue.put(mes)
            except Exception as e:
                continue  # 忽略超时异常，继续运行
    
    def recv_message(self, timeout=None):
        try:
            msg = self.msg_queue.get(True, timeout)
        except queue.Empty:
            return None
        return msg

    def submit(self, func: Callable[[], None]):
        self.executor.submit(func)

    def ack_resp_handler(self, mes: AckMessage):
        if mes.header.tid is None:
            return
        txctx = self.rem_tx(mes.header.tid)
        if txctx is None:
            logging.warning('ack resp non existing tid')
            return
        txctx.response = mes
        txctx.event.set()

    def wait_with_timeout(self, txctx: txContext, timeout: float):
        txctx.event.wait(timeout=timeout)
        return txctx.response

    def transaction_message_handler(self, func, name):
        tid = self.new_tid()
        func(tid)  # TODO 这里可以用线程池发，但考虑到发消息用时很短，暂时不用
        txctx = txContext(tid)
        self.add_tx(tid, txctx)
        
        resp_mes: Optional[AckMessage] = self.wait_with_timeout(txctx, ms2s(self.cfg.tx_timeout))
        if resp_mes is None:
            self.rem_tx(tid)
            logging.warning(f'{name}:{tid} timeout')
            return False
        elif resp_mes.code != 0:
            logging.warning(f'{name}:{tid} failed: {resp_mes.mes}')
            return False
        return True

    def appreg_handler(self, CapID: int, CapVersion: int, CapConfig: int, act: int):
        self.transaction_message_handler(
            lambda tid: self.icp_server.AppMessage(CapID, CapVersion, CapConfig, act, tid), 'APPREG')

    def appreg(self, CapID: int, CapVersion: int, CapConfig: int, act: int):
        self.submit(
            lambda: self.appreg_handler(CapID, CapVersion, CapConfig, act))

    def brocastpub_handler(self, oid: appType.id_t, topic: str, coopMap: bytes, coopMapType: int):
        self.transaction_message_handler(
            lambda tid: self.icp_server.brocastPub(tid, oid, topic, coopMap, coopMapType), 'BROCASTPUB')

    def brocastpub(self, oid: appType.id_t, topic: str, coopMap: bytes, coopMapType: int):
        self.submit(
            lambda: self.brocastpub_handler(oid, topic, coopMap, coopMapType))

    def brocastsub_handler(self, oid: appType.id_t, topic: str, context: str, coopMap: bytes,
                         coopMapType: int, bearCap: int):
        self.transaction_message_handler(
            lambda tid: self.icp_server.brocastSub(tid, oid, topic, context, coopMap, coopMapType, bearCap),
            'BROADCASTSUB')

    def brocastsub(self, oid: appType.id_t, topic: str, context: str, coopMap: bytes,
                  coopMapType: int, bearCap: int):
        self.submit(
            lambda: self.brocastsub_handler(oid, topic, context, coopMap, coopMapType, bearCap))

    def brocastsubnty_handler(self, oid: appType.id_t, did: appType.id_t, topic: str, context: str,
                           coopMap: bytes, coopMapType: int, bearcap: int):
        self.transaction_message_handler(
            lambda tid: self.icp_server.brocastSubnty(tid, oid, did, topic, context, coopMap, coopMapType,
                                                    bearcap), 'BROCASTSUBNTY')

    def brocastsubnty(self, oid: appType.id_t, did: appType.id_t, topic: str, context: str,
                    coopMap: bytes, coopMapType: int, bearcap: int):
        self.submit(
            lambda: self.brocastsubnty_handler(oid, did, topic, context, coopMap, coopMapType, bearcap))

    def subscribe_handler(self, oid: appType.id_t, did: List[str], topic: str, act: int,
                       context: str, coopMap: bytes, coopMapType: int, bearInfo: int):
        self.transaction_message_handler(
            lambda tid: self.icp_server.subMessage(tid, oid, did, topic, act, context, coopMap, coopMapType,
                                                 bearInfo), 'SUBSCRIBE')

    def subscribe(self, oid: appType.id_t, did: List[str], topic: str, act: int,
                context: str, coopMap: bytes, coopMapType: int, bearInfo: int):
        self.submit(
            lambda: self.subscribe_handler(oid, did, topic, act, context, coopMap, coopMapType, bearInfo))

    def notify_handler(self, oid: appType.id_t, did: appType.id_t, topic: str, act: int,
                     context: str, coopMap: bytes, coopMapType: int, bearCap: int):
        self.transaction_message_handler(
            lambda tid: self.icp_server.notifyMessage(tid, oid, did, topic, act, context, coopMap,
                                                    coopMapType, bearCap), 'NOTIFY')

    def notify(self, oid: appType.id_t, did: appType.id_t, topic: str, act: int,
             context: str, coopMap: bytes, coopMapType: int, bearCap: int):
        self.submit(
            lambda: self.notify_handler(oid, did, topic, act, context, coopMap, coopMapType, bearCap))

    def sendfile(self, did: appType.id_t, context: str, rl: int, pt: int, file: str):
        """ 不需要事务 """
        self.submit(lambda: self.icp_server.sendFile(did, context, rl, pt, file))

    def sendreq(self, did: appType.id_t, context: str, rl: int, pt: int, aoi: int, mode: int):
        """ 不需要事务 """
        self.submit(lambda: self.icp_server.streamSendreq(did, context, rl, pt))

    def send(self, sid: appType.sid_t, data: bytes):
        """ 不需要事务 """
        self.submit(lambda: self.icp_server.streamSend(sid, data))

    def sendend(self, did: appType.id_t, cid: appType.cid_t, sid: appType.sid_t):
        """ 不需要事务 """
        self.submit(lambda: self.icp_server.streamSendend(did, cid, sid))