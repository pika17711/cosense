import asyncio

from src.mes.messageHandler import MessageID
"""
协作需求管理模块
Collaborative Requirements Management Module
"""

class CRMMEvent:
    def __init__(self, typ, **args):
        self.type = typ
        self.args = args

class CRMM:
    def __init__(self):
        self.q = asyncio.Queue()
        self.bpub_interval = 0
        self.bpub_last = 0
        self.bsub_interval = 0
        self.bsub_last = 0

    async def add_event(self, e):
        await self.q.put(e)

    async def bsub_timer_event(self):
        while True:
            if self.bsub_interva > 0:
                await asyncio.sleep(self.bsub_interval)
            else:
                await asyncio.sleep(0.5)
            
    async def bpub_timer_event(self):
        while True:
            if self.bsub_interva > 0:
                await asyncio.sleep(self.bpub_interval)
            else:
                await asyncio.sleep(0.5)

    def bsubnty_process(self, event):
        pass

    def 

    async def message_event(self):
        while True:
            event = self.q.get()
            if event.type == MessageID.BROCASTSUBNTY:
                self.bsubnty_process(event)
            elif event.type == MessageID.BROCASTSUB:
                pass
            elif event.type == MessageID.BROCASTPUB:
                pass
            else:
                assert False
