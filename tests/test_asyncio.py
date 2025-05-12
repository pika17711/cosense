import asyncio

class A:
    def __init__(self):
        self.msg_queue = asyncio.Queue()

    async def recv_loop(self):
        while True:
            resp = await self.msg_queue.get()


class transactionHandler:
    def __init__(self, message_queue: asyncio.Queue):
        self.recv_queue = asyncio.Queue()
        self.running = True
        self.message_queue = message_queue

    async def recv_loop(self):
        while self.running:
            resp = await self.recv_queue.get()

class MessageHandler:
    def __init__(self):
        self.msg_queue = asyncio.Queue()
        self.tx_handler: transactionHandler = transactionHandler(self.msg_queue)

    async def recv_loop(self):
        await self.tx_handler.recv_loop()


class CollaborationManager:
    def __init__(self):
        self.message_handler = MessageHandler()

    async def loop(self):
        await self.message_handler.recv_loop()

def main():
    cm = CollaborationManager()
    asyncio.run(cm.loop())