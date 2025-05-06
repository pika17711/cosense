import asyncio
import logging
from config import AppConfig
from mes.messageHandler import MessageHandler

class CollaborationManager:
    def __init__(self):
        self.message_handler = MessageHandler()
        self.broadcastpub_event = asyncio.Event()
        self.broadcastsub_event = asyncio.Event()
        asyncio.create_task(self.broadcastpub_loop())
        asyncio.create_task(self.broadcastsub_loop())

    async def loop(self):
        logging.info("message_handler recv loop start")
        await self.message_handler.recv_loop()

    def broadcastpub_open(self):
        self.broadcastpub_event.set()

    def broadcastpub_close(self):
        self.broadcastpub_event.clear()

    async def broadcastpub_send(self):
        await self.message_handler.broadcastpub_send()

    async def broadcastpub_loop(self):
        while True:
            if self.broadcastpub_event.is_set():
                await self.message_handler.broadcastpub_send()
                await asyncio.sleep(AppConfig.broadcastpub_period)
            else:
                await self.broadcastpub_event.wait()

    def broadcastsub_open(self):
        self.broadcastsub_event.set()

    def broadcastsub_close(self):
        self.broadcastsub_event.clear()

    async def broadcastsub_send(self):
        await self.message_handler.broadcastsub_send()

    async def broadcastsub_loop(self):
        while True:
            if self.broadcastsub_event.is_set():
                await self.message_handler.broadcastsub_send()
                await asyncio.sleep(AppConfig.broadcastsub_period)
            else:
                await self.broadcastsub_event.wait()