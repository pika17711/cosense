from __future__ import annotations

import sys
import os
import logging

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from appConfig import AppConfig
from collaboration.ICP import ICP_init
from collaboration.collaborationRPCServer import CollaborationRPCServerThread
from collaboration.messageRouter import MessageRouter
from collaboration.collaborationTable import CollaborationTable
from collaboration.collaborationService import CollaborationService
from collaboration.transactionHandler import transactionHandler
from collaboration.collaborationManager import CollaborationManager
from perception.perceptionRPCClient import PerceptionRPCClient
from detection.detectionRPCClient import DetectionRPCClient
from utils.othersInfos import OthersInfos

def log_init(cfg: AppConfig):
    """
        日志初始化
    """
    logging.basicConfig(level=logging.DEBUG,
                        filename='collaboration.log',
                        filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("协同模块启动")

def main():
    if len(sys.argv) > 1:
        logging.info("Usage: python qt_main.py")
        exit(-1)

    cfg = AppConfig()
    log_init(cfg)
    icp_client, icp_server = ICP_init(cfg)

    # 全部初始化，依赖注入的思想，方便替换
    perception_client = PerceptionRPCClient(cfg)
    detection_client = DetectionRPCClient()
    ctable = CollaborationTable(cfg)
    tx_handler = transactionHandler(cfg, icp_server, icp_client)
    collaboration_service = CollaborationService(cfg, ctable, perception_client, detection_client, tx_handler)
    message_handler = MessageRouter(cfg, ctable, tx_handler, perception_client, collaboration_service)
    collaboration_manager = CollaborationManager(cfg, ctable, message_handler, perception_client, detection_client, collaboration_service)

    others_infos = OthersInfos(ctable)
    collaboration_rpc_server = CollaborationRPCServerThread(cfg, others_infos)

    tx_handler.start_recv()
    message_handler.start_recv()
    collaboration_manager.start_send_loop()

    collaboration_rpc_server.start()

    try:
        collaboration_manager.command_loop()
    except KeyboardInterrupt:
        print('"接收到 Ctrl + C，程序退出中...')
        tx_handler.close()
        message_handler.close()
        collaboration_manager.close()
        collaboration_rpc_server.close()
        logging.info("接收到 Ctrl + C，程序退出。")

if __name__ == "__main__":
    main()