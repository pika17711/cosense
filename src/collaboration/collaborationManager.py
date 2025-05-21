from __future__ import annotations

import asyncio
import logging
import concurrent.futures
import threading
from time import sleep
from appConfig import AppConfig
from collaboration.message import SubscribeAct
from utils import InfoDTO
from perception.perceptionRPCClient import PerceptionRPCClient

from collaboration.messageHandler import MessageHandler
from collaboration.collaborationTable import CollaborationTable
from collaboration.collaborationService import CollaborationService
from utils.common import ms2s

class CollaborationManager:
    def __init__(self, 
                 cfg: AppConfig,
                 ctable: CollaborationTable,
                 message_handler: MessageHandler,
                 perception_client: PerceptionRPCClient,
                 collaboration_service: CollaborationService):

        self.cfg = cfg
        self.perception_client = perception_client
        self.message_handler = message_handler
        self.ctable = ctable
        self.collaboration_service = collaboration_service

        self.broadcastpub_event = threading.Event()
        self.broadcastsub_event = threading.Event()
        self.subscribed_send_event = threading.Event()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.running = True

        self.broadcastpub_loop_thread = threading.Thread(target=self.broadcastpub_loop, name='broadcastpub_loop', daemon=True)

        self.broadcastsub_loop_thread = threading.Thread(target=self.broadcastsub_loop, name='broadcastsub_loop', daemon=True)

        self.subscribed_send_loop_thread = threading.Thread(target=self.subscribed_send_loop, name='subscribed_send_loop', daemon=True)

    def start_send_loop(self):
        self.broadcastpub_loop_thread.start()
        self.broadcastsub_loop_thread.start()
        self.subscribed_send_loop_thread.start()


    def close(self):
        self.running = False
        self.executor.shutdown()
        if self.broadcastpub_loop_thread.is_alive():
            self.broadcastpub_event.set()
            self.broadcastpub_loop_thread.join(self.cfg.close_timeout)
        if self.broadcastsub_loop_thread.is_alive():
            self.broadcastsub_event.set()
            self.broadcastsub_loop_thread.join(self.cfg.close_timeout)
        if self.subscribed_send_loop_thread.is_alive():
            self.subscribed_send_event.set()
            self.subscribed_send_loop_thread.join(self.cfg.close_timeout)


    def handle_command(self, argv):
        logging.debug(f"输入的命令是: {argv}")
        if len(argv) == 0:
            pass
        elif len(argv) == 1 and argv[0] == 'exit':
            return False
        elif len(argv) == 2 and argv[0] == 'bpub':
            if argv[1] == 'open':
                self.broadcastpub_open()
                print('ok')
            elif argv[1] == 'close':
                self.broadcastpub_close()
                print('ok')
            elif argv[1] == 'send':
                self.collaboration_service.broadcastpub_send()
                print('ok')
            else:
                print('syntax error')
        elif len(argv) == 2 and argv[0] == 'bsub':
            if argv[1] == 'open':
                self.broadcastsub_open()
                print('ok')
            elif argv[1] == 'close':
                self.broadcastsub_close()
                print('ok')
            elif argv[1] == 'send':
                self.collaboration_service.broadcastsub()
                print('ok')
            else:
                print('syntax error')
        elif len(argv) == 2 and argv[0] == 'show':
            if argv[1] == 'subing':
                print([cctx.remote_id() for cctx in self.ctable.get_subscribing()])
            elif argv[1] == 'subed':
                print([cctx.remote_id() for cctx in self.ctable.get_subscribed()])
        elif len(argv) == 2 and argv[0] == 'subscribe':
            self.collaboration_service.subscribe_send(argv[2], SubscribeAct.ACK)
        elif len(argv) == 2 and argv[0] == 'disconnect':
            self.collaboration_service.disconnect(argv[1])
        else:
            print('syntax error')
        return True

    def command_loop(self):
        while self.running:
            try:
                command = input("$ ")
                argv = command.split()
                should_continue = self.handle_command(argv)
                if not should_continue:
                    self.close()
                    break
            except EOFError:
                break

    def broadcastpub_open(self):
        self.broadcastpub_event.set()

    def broadcastpub_close(self):
        self.broadcastpub_event.clear()

    def broadcastpub_send(self):
        self.collaboration_service.broadcastpub_send()

    def broadcastpub_loop(self):
        while self.running:
            if self.broadcastpub_event.is_set():
                self.collaboration_service.broadcastpub_send()
                sleep(ms2s(self.cfg.broadcastpub_period))
            else:
                self.broadcastpub_event.wait()

    def broadcastsub_open(self):
        self.broadcastsub_event.set()

    def broadcastsub_close(self):
        self.broadcastsub_event.clear()

    def broadcastsub_send(self):
        self.collaboration_service.broadcastsub()

    def broadcastsub_loop(self):
        while self.running:
            if self.broadcastsub_event.is_set():
                self.collaboration_service.broadcastsub()
                sleep(ms2s(self.cfg.broadcastsub_period))
            else:
                self.broadcastsub_event.wait()

    def get_all_data(self):
        ts1, pose, velocity, acceleration = self.perception_client.get_my_pva_info()
        ts2, extrinsic_matrix = self.perception_client.get_my_extrinsic_matrix()
        ts3, feat = self.perception_client.get_my_feature()
        infodto = InfoDTO.InfoDTO(1, self.cfg.id, extrinsic_matrix, None, None, feat, ts3, velocity, ts1, pose, ts1, acceleration, ts2, None, None)
        data = InfoDTO.InfoDTOSerializer.serialize(infodto)
        return data

    def subscribed_send_loop(self):
        logging.info("订阅者数据发送循环启动")
        while self.running:
            subeds = self.ctable.get_subscribed()
            data = self.get_all_data()
            logging.info(f"订阅者数据发送, 订阅者列表{[remote_id for remote_id in subeds]}, 发送数据 {len(data)}B")
            if len(subeds) > 0:
                for cctx in subeds:
                    self.executor.submit(self.collaboration_service.send_data, cctx, data)
            self.subscribed_send_event.wait(self.cfg.send_data_period/1000)
            if self.subscribed_send_event.is_set():
                break