from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
from queue import Queue
from threading import Thread
from typing import Callable, Dict

from collaboration.collaborationTable import CollaborationTable
from collaboration.transactionHandler import transactionHandler
from collaboration.collaborationService import CollaborationService

from appConfig import AppConfig
from perception.perceptionRPCClient import PerceptionRPCClient

from collaboration.message import Message, NotifyAct, SubscribeAct
from collaboration.messageID import MessageID
from utils.common import ms2s

class MessageRouter:
    """
        消息路由器

        将对应类型的消息路由到对应的消息处理函数
    """
    def __init__(self, 
                 cfg: AppConfig,
                 ctable: CollaborationTable,
                 tx_handler: transactionHandler,
                 perception_client: PerceptionRPCClient,
                 collaboration_service: CollaborationService):

        self.cfg = cfg
        self.ctable = ctable
        self.tx_handler = tx_handler
        self.collaboration_service = collaboration_service
        self.perception_client= perception_client

        self.max_workers = cfg.message_max_workers
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.running = True
        self.msg_queue = Queue()

        self.route_table: Dict[MessageID, Callable[[Message], None]] = { # type: ignore
            # MessageID.APPRSP: collaboration_service.

            MessageID.BROCASTPUB: collaboration_service.broadcastpub_service,
            MessageID.BROCASTSUB: collaboration_service.broadcastsub_service,
            MessageID.BROCASTSUBNTY: collaboration_service.broadcastsubnty_service,
            MessageID.SUBSCRIBE: collaboration_service.subscribe_service,
            MessageID.NOTIFY: collaboration_service.notify_service,
            
            MessageID.SENDRDY: collaboration_service.sendrdy_service,
            MessageID.RECV: collaboration_service.recv_service,
            MessageID.RECVRDY: collaboration_service.recvrdy_service,
            MessageID.RECVEND: collaboration_service.recvend_service,

            MessageID.RECVFILE: collaboration_service.recvfile_service
        }

        self.recv_thread = Thread(target=self.recv_loop, name='messageHandler recv_loop', daemon=True)

    def start_recv(self):
        self.recv_thread.start()

    def close(self):
        self.running = False
        if self.recv_thread.is_alive():
            self.recv_thread.join(self.cfg.close_timeout)

        self.executor.shutdown()
        for cctx in self.ctable.get_subscribing():
            if cctx.have_sid():
                self.collaboration_service.sendend_send(cctx.remote_id(), cctx.cid, cctx.sid) # type: ignore
            self.collaboration_service.subscribe_send(cctx, SubscribeAct.FIN)

        for subed in self.ctable.get_subscribed():
            self.collaboration_service.notify_send(subed, NotifyAct.FIN)

    def check_expire(self):
        cctxs = self.ctable.get_all_cctx()
        for cctx in cctxs:
            with cctx.lock:
                if cctx.is_expired():
                    self.collaboration_service.cctx_to_closed(cctx)
        
        bcctxs = self.ctable.get_all_bcctx()
        for bcctx in bcctxs:
            with bcctx.lock:
                if bcctx.is_expired():
                    self.collaboration_service.bcctx_to_closed(bcctx)

    def recv_loop(self):
        while self.running:
            msg = self.tx_handler.recv_message(0.1)  # timeout for check self.running
            if msg is None:
                continue
            self.dispatch_message(msg)
            self.check_expire()

    def dispatch_message(self, msg: Message):
        mid = msg.header.mid
        if mid in self.route_table:
            self.route_table[mid](msg)  # TODO 使用线程池
        else:
            logging.warning(f"Unhandled message type: {MessageID.get_name(mid.value)}")