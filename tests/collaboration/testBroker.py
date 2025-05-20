import datetime
import hashlib
import os
import sys
import zmq
import json
import logging
from typing import Dict, List, Set, Tuple
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
from src.collaboration.messageID import MessageID
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestBroker:
    def __init__(self, server_bind_addr: str = "tcp://*:5555", client_bind_addr: str = "tcp://*:5556"):
        """
        初始化Broker
        :param server_bind_addr: 服务器连接的绑定地址（ICPServer连接此处）
        :param client_bind_addr: 客户端连接的绑定地址（ICPClient连接此处）
        """
        self.context = zmq.Context()
        
        # 用于服务器注册和消息发送的ROUTER套接字（服务器作为客户端连接）
        self.server_socket = self.context.socket(zmq.ROUTER)
        self.server_socket.bind(server_bind_addr)
        logging.info(f"Server socket bound to {server_bind_addr}")
        
        # 用于客户端订阅和消息接收的ROUTER套接字（客户端作为客户端连接）
        self.client_socket = self.context.socket(zmq.ROUTER)
        self.client_socket.bind(client_bind_addr)
        logging.info(f"Client socket bound to {client_bind_addr}")
        
        # id -> identity
        self.server_registry: Dict[str, bytes] = {}
        # id -> identity
        self.client_registry: Dict[str, bytes] = {}

        # sid -> (id, context)
        self.sid_table: Dict[str, Tuple] = {}

        self.poller = zmq.Poller()
        self.poller.register(self.server_socket, zmq.POLLIN)
        self.poller.register(self.client_socket, zmq.POLLIN)

    def run(self):
        """
        启动Broker主循环
        """
        logging.info("Broker started, waiting for connections...")
        while True:
            socks = dict(self.poller.poll())
            
            # 处理服务器连接或消息
            if self.server_socket in socks:
                self.handle_server_message()
            
            # 处理客户端连接或消息
            if self.client_socket in socks:
                self.handle_client_message()

    def handle_server_message(self):
        """
        处理服务器端的连接和消息
        """
        try:
            parts = self.server_socket.recv_multipart()
            identity = parts[0]
            msg = json.loads(parts[-1].decode('utf-8'))
            logging.info(f"message: {msg}")

            if msg.get("type") == "register":
                # 服务器注册消息处理
                app_id = msg["app_id"]
                oid = msg["oid"]
                self.server_registry[oid] = identity
                logging.info(f"Server registered: app_id={app_id}, oid={oid}")
            elif msg.get("mid") == MessageID.BROCASTPUB.value:  # 广播发布消息
                self.broadcast_message_to_all(parts, source_type="server")
            elif msg.get("mid") == MessageID.BROCASTSUB.value:
                self.broadcast_message_to_all(parts, source_type="server")
            elif msg.get("mid") == MessageID.SENDREQ.value:
                self.handle_sendreq_message(parts)
            else:
                # 常规消息处理（按oid转发）
                self.forward_message(parts, source_type="server")
            
        except Exception as e:
            logging.error(f"Error processing server message: {e}")

    def gen_sid():
        return hashlib.md5(str(datetime.datetime.now()))

    def handle_sendreq_message(self, parts):
        sid = self.gen_sid()
        self.sid_table[sid] = (parts["oid"], parts.get('context'))
        msg = {'mid': MessageID.SENDRDY.value, 'sid': sid}
        part = json.dumps(msg).encode()
        self.send_message(parts['did'], part)

    def handle_client_message(self):
        """
        处理客户端的连接和消息
        """
        try:
            parts = self.client_socket.recv_multipart()
            identity = parts[0]
            msg = json.loads(parts[-1].decode('utf-8'))

            if msg.get("type") == "subscribe":
                # 客户端订阅处理
                oid = msg["oid"]
                self.client_registry[oid] = identity
                logging.info(f"Client subscribed: oid={oid}")
            else:
                # 客户端请求处理（可选）
                logging.warning(f"Unexpected client message: {msg}")
                
        except Exception as e:
            logging.error(f"Error processing client message: {e}")

    def broadcast_message_to_all(self, parts: List[bytes], source_type: str):
        """
        广播消息给所有注册的客户端
        :param parts: 原始消息部件
        :param source_type: 消息来源类型
        """
        try:
            # 提取消息内容
            msg = json.loads(parts[-1].decode('utf-8'))
            
            # 获取所有已注册的客户端
            clients = [entry for entry in self.client_registry.values()]
            
            if not clients:
                logging.info("No clients to broadcast message to")
                return

            # 广播消息给所有客户端
            for client in clients:
                target_identity = client
                forwarded_parts = [target_identity, b"", parts[-1]]
                self.client_socket.send_multipart(forwarded_parts)

            logging.info(f"Broadcast message from {source_type} sent to {len(clients)} clients")
            
        except Exception as e:
            logging.error(f"Error broadcasting message: {e}")

    def send_message(self, target_id, part):            
        # 查找目标客户端或服务器
        target_entry = self.client_registry.get(target_id)

        if not target_entry:
            logging.warning(f"No recipient found for id={target_id}")
            return
            
        target_socket = self.client_socket
        target_identity = target_entry

        # 移除原始identity，添加目标identity
        forwarded_parts = [target_identity, b"", part]
        target_socket.send_multipart(forwarded_parts)

    def forward_message(self, parts: List[bytes]):
        """
        按消息中的oid字段转发消息
        :param parts: 原始消息部件（包含identity和消息内容）
        :param source_type: 消息来源类型（server/client）
        """
        try:
            msg = json.loads(parts[-1].decode('utf-8'))
            target_id = msg["msg"].get("did")  # 从消息体中提取目标oid
            
            self.send_message(target_id, parts[-1])
            
            logging.info(f"Forwarded {MessageID(msg['mid']).name} from to oid={target_id}")
            
        except Exception as e:
            logging.error(f"Error forwarding message: {e}")