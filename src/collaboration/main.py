from __future__ import annotations

import sys
import os
import threading

from appConfig import AppConfig

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from collaboration.ICP import ICP_init
import logging
from collaboration.collaborationRPCServer import CollaborationRPCServerThread, SharedOthersInfo
from collaboration.messageHandlerSync import MessageHandlerSync
from collaboration.collaborationTable import CollaborationTable
from collaboration.collaborationService import CollaborationService
from collaboration.transactionHandlerSync import transactionHandlerSync
from collaboration.collaborationManager import CollaborationManager
from perception.perceptionRPCClient import PerceptionRPCClient

def log_init(cfg: AppConfig):
    logging.basicConfig(level=logging.DEBUG,
                        filename='collaboration.log',
                        filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("协同模块启动")

def main():
    if len(sys.argv) > 1:
        logging.info("Usage: python main.py")
        exit(-1)

    cfg = AppConfig()
    log_init(cfg)
    icp_client, icp_server = ICP_init(cfg)

    perception_client = PerceptionRPCClient(cfg)
    ctable = CollaborationTable(cfg)
    tx_handler = transactionHandlerSync(cfg, icp_server, icp_client)
    collaboration_service = CollaborationService(cfg, ctable, perception_client, tx_handler)
    message_handler = MessageHandlerSync(cfg, ctable, tx_handler, perception_client, collaboration_service)
    collaboration_manager = CollaborationManager(cfg, ctable, message_handler, perception_client, collaboration_service)

    shared_other_info = SharedOthersInfo(ctable)
    collaboration_rpc_server = CollaborationRPCServerThread(cfg, shared_other_info)

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