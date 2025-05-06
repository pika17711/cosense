import asyncio
from enum import IntEnum
import logging
import random
from typing import Dict, List
from config import AppConfig
from mes.messageID import MessageID
from mes.message import AckMessage, Message, AppRspMessage
from ICP.ICP import icp_client, icp_server
import concurrent.futures

"""transactionHandler"""
class txContext:
    def __init__(self, tid):
        self.tid = tid
        self.queue = asyncio.Queue(1)

class transactionHandler:
    def __init__(self, message_queue: asyncio.Queue):
        self.recv_queue = asyncio.Queue()
        self.tid_counter = random.randint(0, 1 << 20)
        self.tx_table: Dict[int, txContext] = dict() # tid -> txContext
        self.icp_server = icp_server
        self.icp_client = icp_client
        self.running = True
        self.message_queue = message_queue
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def new_tid(self):
        self.tid_counter += 1
        return self.tid_counter

    def add_tx(self, tid, txctx):
        self.tx_table[tid] = txctx

    async def recv_icp(self):
        loop = asyncio.get_running_loop()
        while self.running:
            future = loop.run_in_executor(self.executor, self.icp_client.recv_message)
            result = await future
            await self.recv_queue.put(result)

    async def recv_loop(self):
        asyncio.create_task(self.recv_icp())
        while self.running:
            resp = await self.recv_queue.get()
            mes = Message.parse(resp)
            if mes.header.mid == MessageID.ACK:
                await self.ack_resp_handler(mes)
            else:
                if MessageID.is_control(mes.header.mid):
                    # TODO 发送ACK消息给通信模块，告诉它已经收到，目前ICPServer无此方法
                    pass
                await self.message_queue.put(mes)
    
    async def submit_transaction(self, coro):
        task = asyncio.create_task(coro)
        return await task

    async def ack_resp_handler(self, mes: Message):
        try:
            await self.tx_table[mes.header.tid].queue.put(mes)
            self.tx_table.pop(mes.header.tid)
        except KeyError:
            logging.warning('ack resp non existing tid')

    async def wait_with_timeout(self, queue: asyncio.Queue, timeout: int):
        try:
            return await asyncio.wait_for(queue.get(), timeout=timeout/1000)
        except asyncio.TimeoutError:
            return None
        
    async def xxx_handler(self, func, name):
        tid = self.new_tid()
        func(tid)
        txctx = txContext(tid)
        resp_mes: AckMessage = await self.wait_with_timeout(txctx.queue, self.tx_timeout/1000)
        if resp_mes is None:
            try:
                self.tx_table.pop(tid)
            except KeyError:
                assert False
            logging.warning(f'{name}:{tid} timeout')
            return False
        elif resp_mes.code != 0:
            logging.warning(f'{name}:{tid} failed: {resp_mes.mes}')
            return False
        return True

    async def appreg_handler(self, CapID:int, CapVersion:int, CapConfig:int, act:int) -> bool:
        return self.xxx_handler(lambda tid: self.icp_server.AppMessage(CapID, CapVersion, CapConfig, act, tid), 'APPREG')

    async def appreg(self, CapID:int, CapVersion:int, CapConfig:int, act:int):
        return self.submit_transaction(self.appreg_handler(CapID, CapVersion, CapConfig, act))

    async def brocastpub_handler(self, oid: str, topic: int, coopMap: str, coopMapType: int) -> bool:
        return self.xxx_handler(lambda tid: self.icp_server.brocastPub(tid, oid, topic, coopMap, coopMapType), 'BROCASTPUB')

    async def brocastpub(self, oid: str, topic: int, coopMap: str, coopMapType: int):
        return self.submit_transaction(self.brocastpub_handler(oid, topic, coopMap, coopMapType))

    async def brocastsub_handler(self, oid: str, topic: int, context: str, coopMap: str, 
                                 coopMapType: int, bearCap: int) -> bool:
        return self.xxx_handler(lambda tid: self.icp_server.brocastSub(tid, oid, topic, context, coopMap, coopMapType, bearCap), 'BROADCASTSUB')

    async def brocastsub(self, oid: str, topic: int, context: str, coopMap: str, 
                         coopMapType: int, bearCap: int):
        return self.submit_transaction(self.brocastsub_handler(oid, topic, context, coopMap, coopMapType, bearCap))

    async def brocastsubnty_handler(self, oid: str, did: str, topic: int, context: str, 
                                  coopMap: str, coopMapType: int, bearcap: int) -> bool:
        return self.xxx_handler(lambda tid: self.icp_server.brocastSubnty(tid, oid, did, topic, context, coopMap, coopMapType, bearcap), 'BROCASTSUBNTY')

    async def brocastsubnty(self, oid: str, did: str, topic: int, context: str, 
                          coopMap: str, coopMapType: int, bearcap: int):
        return self.submit_transaction(self.brocastsubnty_handler(oid, did, topic, context, coopMap, coopMapType, bearcap))

    async def subscribe_handler(self, oid: str, did: List[str], topic: int, act: int, 
                              context: str, coopMap: str, coopMapType: int, bearInfo: int) -> bool:
        return self.xxx_handler(lambda tid: self.icp_server.subMessage(tid, oid, did, topic, act, context, coopMap, coopMapType, bearInfo), 'SUBSCRIBE')

    async def subscribe(self, oid: str, did: List[str], topic: int, act: int, 
                      context: str, coopMap: str, coopMapType: int, bearInfo: int):
        return self.submit_transaction(self.subscribe_handler(oid, did, topic, act, context, coopMap, coopMapType, bearInfo))

    async def notify_handler(self, oid: str, did: str, topic: int, act: int, 
                           context: str, coopMap: str, coopMapType: int, bearCap: int) -> bool:
        return self.xxx_handler(lambda tid: self.icp_server.notifyMessage(tid, oid, did, topic, act, context, coopMap, coopMapType, bearCap), 'NOTIFY')

    async def notify(self, oid: str, did: str, topic: int, act: int, 
                   context: str, coopMap: str, coopMapType: int, bearCap: int):
        return self.submit_transaction(self.notify_handler(oid, did, topic, act, context, coopMap, coopMapType, bearCap))

    async def sendfile(self, did: str, context: str, rl: int, pt: int, file: str):
        """ 不需要事务 """
        self.icp_server.sendFile(did, context, rl, pt, file)