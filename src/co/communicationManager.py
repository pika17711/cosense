
import asyncio
from client import agent_client
from server import agent_server
from commDecisionEngine import CommDecisionEngine
from utils import mstime

class CommunicationManager:
    """通信管理核心类"""
    def __init__(self):
        self.decision_engine = CommDecisionEngine()

    async def execute_communication(self, result):
        comm_masks_dict = result['comm_masks']
        fused_feat = result['fused_feat']

        comm_targets = agent_server.get_subscribed()
        for target in comm_targets:
            agent_client.pub(target, {'feat': fused_feat, 'comm_mask': comm_masks_dict[target]})