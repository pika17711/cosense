import zmq
import json
import sys
import os

from src.config import AppConfig

class ICPClient:
    def __init__(self,topic = ""):
        """
        初始化 ICPClient 类，连接到指定端口。
        """
        self.port = AppConfig.zmq_in_port
        self.ip = AppConfig.zmq_in_host
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{self.ip}:{self.port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        #print(f"Client connected to port {self.port}")
        
    def recv_message(self):
        """
        接收消息方法：支持带 topic 和不带 topic 的情况
        """
        message = self.socket.recv_string()
        try:
            # 判断是否可能包含 topic（按第一个空格拆分）
            topic, json_part = message.split(" ", 1)
            parsed_message = json.loads(json_part)
            return parsed_message
        except json.JSONDecodeError:
            print(f"[!] Failed to decode message: {message}")
            return None
        

icp_client = ICPClient(AppConfig.topic)