
from ast import Dict
import asyncio
import logging
from config import CONFIG
from ICPHandler import ICP_handler, ICPMessage

defaultSubMessage = ICPMessage('', '', 0, ICPMessage.MTSUB, None, None, None, None, CONFIG['id'], '?')
defaultPubMessage = ICPMessage('', '', 0, ICPMessage.MTPUB, None, None, None, None, CONFIG['id'], '?')
defaultEchoMessage = ICPMessage('', '', 0, ICPMessage.MTECHO, None, None, None, None, CONFIG['id'], '?')

class AgentClientConnection:
    PUBSTATEACK = 1  # 开始推送/订购确认
    PUBSTATEPRO = 2  # 推送过程中
    PUBSTATEFIN = 3  # 推送结束/主动结束订购

    def __init__(self) -> None:
        self.id = None
        self.peer_id = None
        self.source_id = None
        self.agent = None
        self.cached_req = None

        self.pub_state = self.PUBSTATEACK
        self.sub_state = None  # TODO

    async def echo(self):
        msg = defaultPubMessage
        msg.source_id = self.agent.id
        await ICP_handler.echo(msg)

    async def unsub(self):
        msg = defaultSubMessage
        msg.operator = ICPMessage.OPSUBUNDO
        msg.peer_id = self.peer_id
        msg.source_id = self.source_id
        await ICP_handler.sub_undo(msg)
        self.agent.remove_conn(self.id)

    async def sub(self):
        msg = defaultSubMessage
        msg.operator = ICPMessage.OPSUBDO
        msg.peer_id = self.peer_id
        msg.source_id = self.source_id
        await ICP_handler.sub_do(msg)

    async def pub(self, data):
        msg = defaultPubMessage
        msg.data = data
        if self.pub_state == self.PUBSTATEACK:
            await ICP_handler.pub_ack(msg)
            self.pub_state = self.PUBSTATEPRO
        elif self.pub_state == self.PUBSTATEPRO:
            await ICP_handler.pub_pro(msg)

    async def unpub(self, data):
        msg = defaultPubMessage
        msg.data = data
        if self.pub_state == self.PUBSTATEPRO:
            await ICP_handler.pub_fin(msg)
            self.pub_state = self.PUBSTATEACK
        else:
            await ICP_handler.pub_fin(msg)
            self.pub_state = self.PUBSTATEACK
            logging.debug(f'conn {self.id} protocal pub state abnormal')

        self.agent.remove_conn(self.id)

class AgentClient:
    def __init__(self) -> None:
        self.id = CONFIG['id']

    async def sub(self, target):
        msg = defaultSubMessage
        msg.peer_id = target
        msg.source_id = CONFIG['id']
        await ICP_handler.sub_do(msg)

    async def unsub(self, target):
        msg = defaultSubMessage
        msg.peer_id = target
        msg.source_id = CONFIG['id']
        await ICP_handler.sub_undo(msg)

    async def pub(self, target, data):
        msg = defaultPubMessage
        msg.peer_id = target
        msg.source_id = CONFIG['id']
        await ICP_handler.pub_pro(msg)

    async def echo(self, target):
        # useless
        pass

agent_client = AgentClient()