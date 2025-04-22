import asyncio
import json
from typing import Dict
from co.ICPClient import icp_client
from co.ICPserver import icp_server
import zmq
from zmq.asyncio import Context
from src.config import CONFIG, AppConfig

from enum import IntEnum, auto

from utils.common import mstime
from src.mes.messageHandler import MessageHandler

class CollaborationManager:
    def __init__(self):
        self.message_handler = MessageHandler()
        asyncio.create_task(self.message_handler.recv_loop())
