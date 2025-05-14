from __future__ import annotations

import sys
import os

from collaboration.ICP import ICP_init
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import logging
from collaboration.collaborationConfig import CollaborationConfig
from collaboration.messageHandlerSync import MessageHandlerSync
from collaboration.collaborationTable import CollaborationTable
from collaboration.collaborationService import CollaborationService
from collaboration.transactionHandlerSync import transactionHandlerSync
from collaboration.CollaborationManager import CollaborationManager
from perception.perception_client import PerceptionClient

def log_init(cfg: CollaborationConfig):
    logging.basicConfig(level=logging.DEBUG,
                        filename='collaboration.log',
                        filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("协同模块启动")

def main():
    if len(sys.argv) > 1:
        logging.info("Usage: python main.py")
        exit(-1)

    cfg = CollaborationConfig()
    log_init(cfg)
    icp_client, icp_server = ICP_init(cfg)

    perception_client = PerceptionClient()
    ctable = CollaborationTable(cfg)
    tx_handler = transactionHandlerSync(cfg, icp_server, icp_client)
    collaboration_service = CollaborationService(cfg, ctable, perception_client, tx_handler)
    message_handler = MessageHandlerSync(cfg, ctable, tx_handler, perception_client, collaboration_service)
    collaborationManager = CollaborationManager(cfg, ctable, message_handler, perception_client, collaboration_service)

    try:
        collaborationManager.command_loop()
    except KeyboardInterrupt:
        print('"接收到 Ctrl + C，程序退出中...')
        tx_handler.close()
        message_handler.close()
        collaborationManager.close()
        logging.info("接收到 Ctrl + C，程序退出。")

if __name__ == "__main__":
    main()