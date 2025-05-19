from __future__ import annotations

import random
import string
import sys
import os
import logging


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.appConfig import AppConfig
from src.collaboration.collaborationRPCServer import CollaborationRPCServerThread, SharedOthersInfo
from collaboration.messageHandler import MessageHandler
from src.collaboration.collaborationTable import CollaborationTable
from src.collaboration.collaborationService import CollaborationService
from collaboration.transactionHandler import transactionHandler
from src.collaboration.collaborationManager import CollaborationManager
from src.perception.perceptionRPCClient import PerceptionRPCClient
from tests.collaboration.testICPServer import TestICPServer
from tests.collaboration.TestICPClient import TestICPClient

from src.utils.common import mstime

def log_init(cfg: AppConfig):
    path = f'tests/collaboration/tmp{str(mstime())[:8]}'
    try:
        os.mkdir(path)
    except Exception:
        pass
    logging.basicConfig(level=logging.DEBUG,
                        filename=f'{path}/test_collaboration_{cfg.id}.log',
                        filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("协同模块启动")

def ICP_init(cfg: AppConfig):
    icp_server = TestICPServer(cfg.app_id, cfg.id)
    icp_client = TestICPClient(cfg.id)
    return icp_client, icp_server

def main():
    oid = random.choice(string.ascii_letters) * 5

    cfg = AppConfig()
    cfg.id = oid

    log_init(cfg)
    icp_client, icp_server = ICP_init(cfg)

    perception_client = PerceptionRPCClient(cfg)
    ctable = CollaborationTable(cfg)
    tx_handler = transactionHandler(cfg, icp_server, icp_client)
    collaboration_service = CollaborationService(cfg, ctable, perception_client, tx_handler)
    message_handler = MessageHandler(cfg, ctable, tx_handler, perception_client, collaboration_service)
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