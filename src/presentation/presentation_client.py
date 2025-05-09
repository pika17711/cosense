import logging
import grpc
import numpy as np

from rpc import Service_pb2
from rpc import Service_pb2_grpc


class PresentationClient:                   # 信息呈现子系统的Client类，用于向信息呈现子系统的服务器请求服务
    def __init__(self):
        presentation_channel = grpc.insecure_channel('localhost:50054', options=[                 # 与信息呈现子系统建立连接
            ('grpc.max_send_message_length', 64 * 1024 * 1024),                     # 设置gRPC 消息的最大发送和接收大小为64MB
            ('grpc.max_receive_message_length', 64 * 1024 * 1024)])
        self.__presentation_stub = Service_pb2_grpc.PerceptionServiceStub(presentation_channel)
