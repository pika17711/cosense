import asyncio
from copy import copy
from typing import Dict
import json
import numpy as np
from collections import namedtuple

import logging
from client import agent_client

TimestampedPointCloud = namedtuple('TimestampedPointCloud', ['pointcloud', 'timestamp'])

from config import CONFIG
from ICPHandler import ICP_handler, ICPMessage
from utils import AsyncVariable, calculate_overlap, mstime

class AgentServerConnection:
    def __init__(self) -> None:
        self.id = None
        self.peer_id = None
        self.source_id = None
        self.server: AgentServer = None
        self.close = False

        self.cached_echo_msg = AsyncVariable()
        self.cached_pub_msg = AsyncVariable()

    async def get_cached_data(self):
        echo_msg: ICPMessage = await self.cached_echo_msg.get()
        pub_msg: ICPMessage = await self.cached_pub_msg.get()
        if echo_msg is None or pub_msg is None:
            return None

        data = {}
        data['params'] = echo_msg.data # ego_pose, lidar_pose, transformation_matrix, speed
        data.update(pub_msg.data)  # feat, comm_mask
        return data

    def close_conn(self):
        self.server.conns.pop(self.id)

    async def process_echo(self, msg: ICPMessage):
        cached_echo_msg = self.cached_echo_msg.get()
        await self.cached_echo_msg.set(msg)
        
        # 如果第一次echo、位置变动很大、时间很长，向对面发echo 
        # TODO
        await agent_client.pub(msg.source_id, {"lidar": []})

    async def process_sub(self, msg: ICPMessage):
        if msg.is_sub_do():
            self.server.add_sub(self.id)
        else:
            self.server.remove_sub(self.id)

    async def process_pub(self, msg: ICPMessage):
        # 判断是否继续订阅该车
        ego_comm_mask = self.server.get_ego_comm_mask()
        if ego_comm_mask != None:
            if calculate_overlap(ego_comm_mask, msg.data['comm_mask']) < CONFIG['processing']['overlap_threshold']:
                agent_client.unsub(msg.peer_id)

        self.cached_pub_msg.set(msg)

    async def dispatch_message(self, msg: ICPMessage):
        if msg.is_echo():
            await self.process_echo(msg)
        elif msg.is_pub():
            await self.process_pub(msg)
        elif msg.is_sub():
            await self.process_sub(msg)
        else:
            logging.critical('收到了非ECHO PUB SUB的消息')

    async def recv(self, msg):
        await self.dispatch_message(msg)

class AgentServer:
    def __init__(self):
        self.id = CONFIG['id']

        self.ego_pcd = AsyncVariable()

        self.ego_feat = AsyncVariable()
        self.ego_comm_mask = AsyncVariable()

        self.conns: Dict[str, AgentServerConnection] = {} # surrounding vehicle id

        self.subscribed_conns_id = set()

    async def get_neighbors_cached_data(self) -> Dict:
        data = {}
        for neighbor in self.get_neighbors():
            data[neighbor.id] = await neighbor.get_cached_data()
        return data

    def get_neighbors(self):
        return self.conns.values()
    
    def get_subscribed(self):
        return self.subscribed_conns_id

    def add_sub(self, cid):
        self.subscribed_conns_id.add(cid)

    async def clean_timeout_conn(self):
        for conn in self.conns.values():
            if mstime() - await conn.cached_echo_msg.get_timestamp() > CONFIG['keepalive_timeout']:
                self.remove_conn(conn.id)

    def remove_sub(self, cid):
        self.subscribed_conns_id.remove(cid)

    def is_new_conn(self, msg: ICPMessage):
        return msg.message_type == ICPMessage.MTECHO and msg.peer_id not in self.conns

    async def recv_loop(self):
        while True:
            msg: ICPMessage = await ICP_handler.wait_message()
            if self.is_new_conn(msg):
                self.add_conn(msg.source_id)  # 用对方id作conn id
            if msg.source_id in self.conns:
                self.active_conn(msg.source_id, msg)

    def remove_conn(self, cid):
        self.remove_sub()
        self.conns.pop(cid)

    def add_conn(self, cid):
        ac = AgentServerConnection()
        ac.peer_id = cid
        ac.server = self
        self.conns[cid] = ac

    def active_conn(self, cid, msg):
        asyncio.create_task(self.conns[cid].recv(msg))

    async def set_ego_feat(self, v):
        await self.ego_feat.set(v)

    async def wait_ego_feat(self, timeout=None):
        return await self.ego_feat.wait(timeout)

    async def wait_ego_comm_mask(self, timeout=None):
        return await self.ego_comm_mask.wait(timeout)

    async def get_ego_comm_mask(self):
        return await self.ego_comm_mask.get()

    async def set_ego_pcd(self, v):
        await self.ego_pcd.set(v)

    async def wait_ego_pcd(self):
        await self.ego_pcd.wait()

agent_server = AgentServer()