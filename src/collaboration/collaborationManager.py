from __future__ import annotations

import asyncio
import logging
import concurrent.futures
import threading
import time
from time import sleep

import numpy as np

from appConfig import AppConfig
from collaboration.message import SubscribeAct
from utils import InfoDTO
from perception.perceptionRPCClient import PerceptionRPCClient
from detection.detectionRPCClient import DetectionRPCClient

from collaboration.coopMap import CoopMap, CoopMapType
from collaboration.messageRouter import MessageRouter
from collaboration.collaborationTable import CollaborationTable
from collaboration.collaborationService import CollaborationService
from utils.common import ms2s

class CollaborationManager:
    """
        collaboration子系统管理器，负责
            1. 管理广播推送发送
            2. 管理广播订阅发送
            3. 接收命令行的命令
            4. 向订阅者发送自车数据
    """

    def __init__(self, 
                 cfg: AppConfig,
                 ctable: CollaborationTable,
                 message_handler: MessageRouter,
                 perception_client: PerceptionRPCClient,
                 detection_client: DetectionRPCClient,
                 collaboration_service: CollaborationService):

        self.cfg = cfg
        self.perception_client = perception_client
        self.detection_client = detection_client
        self.message_handler = message_handler
        self.ctable = ctable
        self.collaboration_service = collaboration_service

        self.broadcastpub_event = threading.Event()
        self.broadcastsub_event = threading.Event()
        self.subscribed_send_event = threading.Event()
        self.subscribing_update_event = threading.Event()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.running = True

        self.broadcastpub_loop_thread = threading.Thread(target=self.broadcastpub_loop, name='broadcastpub_loop', daemon=True)

        self.broadcastsub_loop_thread = threading.Thread(target=self.broadcastsub_loop, name='broadcastsub_loop', daemon=True)

        self.subscribed_send_loop_thread = threading.Thread(target=self.subscribed_send_loop, name='subscribed_send_loop', daemon=True)

        self.subscribing_update_loop_thread = threading.Thread(target=self.subscribing_update_loop, name='subscribing_update_loop', daemon=True)

    def start_send_loop(self):
        self.broadcastpub_loop_thread.start()
        self.broadcastsub_loop_thread.start()
        self.subscribed_send_loop_thread.start()
        self.subscribing_update_loop_thread.start()


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
        if self.subscribing_update_loop_thread.is_alive():
            self.subscribing_update_event.set()
            self.subscribing_update_loop_thread.join(self.cfg.close_timeout)


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
                print([subed.remote_id() for subed in self.ctable.get_subscribed()])
            else:
                print('syntax error')
        elif len(argv) == 2 and argv[0] == 'subscribe':
            self.collaboration_service.subscribe_send(argv[1], SubscribeAct.ACKUPD)
            print('ok')
        elif len(argv) == 2 and argv[0] == 'disconnect':
            self.collaboration_service.disconnect(argv[1])
            print('ok')
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

    def get_all_data(self, coopmap: CoopMap):
        lidar_pose, ts_lidar_pose, speed, ts_spd, acceleration, ts_acc = self.perception_client.get_my_pva()
        # my_extrinsic_matrix, ts_extrinsic_matrix = self.perception_client.get_my_extrinsic_matrix()
        # lidar_poses = {'other': {'lidar_pose': coopmap.lidar_pose, 'ts_lidar_pose': int(time.time())}}
        # projected_spatial_feature = self.detection_client.lidar_poses_to_projected_spatial_features(lidar_poses)
        comm_masked_feature, comm_mask, ts_feature = self.detection_client.request_map_to_projected_comm_masked_feature(
            coopmap.lidar_pose, coopmap.map, int(time.time()))

        if comm_masked_feature is None:
            return None

        # TODO:
        comm_mask = comm_mask.copy().astype(np.int8)
        packed_comm_mask = np.packbits(comm_mask, axis=-1)
        binary_comm_mask = packed_comm_mask.tobytes()


        if self.cfg.collaboration_pcd_debug:
            pcd, ts_pcd = self.perception_client.get_my_pcd()
        else:
            pcd = None
            ts_pcd = None

        infodto = InfoDTO.InfoDTO(type=1,
                                  id=self.cfg.id,
                                  lidar2world=None,
                                  camera2world=None,
                                  camera_intrinsic=None,
                                  feat={'spatial_feature': comm_masked_feature,
                                        'comm_mask': binary_comm_mask},
                                  ts_feat=ts_feature,
                                  speed=speed,
                                  ts_spd=ts_spd,
                                  lidar_pos=lidar_pose,
                                  ts_lidar_pos=ts_lidar_pose,
                                  acc=acceleration,
                                  ts_acc=ts_acc,
                                  pcd=pcd,
                                  ts_pcd=ts_pcd)
        data = InfoDTO.InfoDTOSerializer.serialize(infodto)
        # data = InfoDTO.InfoDTOSerializer.serialize_to_str(infodto)
        return data

    def subscribed_send_loop(self):
        logging.info("订阅者数据发送循环启动")
        while self.running:
            subeds = self.ctable.get_subscribed()
            if len(subeds) > 0:
                logging.info(f"订阅者数据发送, 订阅者列表{[cctx.remote_id() for cctx in subeds]}")
                for cctx in subeds:
                    coopmap = self.ctable.get_coopmap(cctx.remote_id())
                    if self.cfg.collaboration_no_coopmap_debug:
                        coopmap.map = np.ones((1, 1, 48, 176), dtype=np.float32)
                    if self.cfg.collaboration_request_map_debug:
                        coopmap.map[:] = 1
                    data = self.get_all_data(coopmap)
                    if data is not None:
                        self.executor.submit(self.collaboration_service.send_data, cctx, data)
                    else:
                        logging.info(f'data is None, 取消对订阅者{cctx.remote_id()}的发送')
                    # self.executor.submit(self.collaboration_service.sendend_send(cctx.remote_id(), cctx.cid, cctx.sid))
            self.subscribed_send_event.wait(ms2s(self.cfg.send_data_period))
            if self.subscribed_send_event.is_set():
                break

    def subscribing_update_loop(self):
        logging.info("被订阅者数据更新循环启动")
        while self.running:
            subings = self.ctable.get_subscribing()
            if len(subings) > 0:
                logging.info(f"被订阅者数据更新, 被订阅者列表{[cctx.remote_id() for cctx in subings]}")
                for cctx in subings:
                    with cctx.lock:
                        self.collaboration_service.subscribe_send(cctx, SubscribeAct.ACKUPD)
                        # self.collaboration_service.cctx_to_waitnty(cctx)
            self.subscribing_update_event.wait(ms2s(self.cfg.update_data_period))
            if self.subscribing_update_event.is_set():
                break