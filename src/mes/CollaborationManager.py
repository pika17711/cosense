import asyncio
import json
from typing import Dict
import zmq
from zmq.asyncio import Context
from config import AppConfig

from enum import IntEnum, auto

from utils.common import mstime
from mes.messageHandler import MessageHandler

class CollaborationManager:
    def __init__(self):
        self.message_handler = MessageHandler()

    async def loop(self):
        await self.message_handler.recv_loop()