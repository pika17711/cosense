import logging
import grpc
import numpy as np

from rpc import Service_pb2
from rpc import Service_pb2_grpc
from utils.rpc_utils import protobuf_to_dict

from appConfig import AppConfig


class CollaborationRPCClient:  # 协同感知子系统的Client类，用于向协同感知子系统的服务器请求服务
    def __init__(self, cfg: AppConfig):
        collaboration_channel = grpc.insecure_channel('localhost:50052', options=[  # 与协同感知子系统的服务器建立连接
            ('grpc.max_send_message_length', 64 * 1024 * 1024),  # 设置gRPC 消息的最大发送和接收大小为64MB
            ('grpc.max_receive_message_length', 64 * 1024 * 1024)])
        self.__collaboration_stub = Service_pb2_grpc.CollaborationServiceStub(collaboration_channel)
        self.cfg = cfg

    def get_others_infos(self):  # 从协同感知子系统获取所有他车信息
        if self.cfg.rpc_collaboration_client_debug:
            return {'DEBUG': {'lidar_pose': np.ones((1, 3)),
                              'ts_lidar_pose': 1,
                              'speed': np.ones((1, 3)),
                              'ts_spd': 1,
                              'acceleration': np.ones((1, 3)),
                              'ts_acceleration': 1,
                              'feature': np.ones((1, 3)),
                              'ts_feature': 1}}

        try:
            response = self.__collaboration_stub.GetOthersInfos(Service_pb2.Empty(), timeout=100)  # 请求协同感知子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC get_others_infos failed: code={e.code()}")  # 记录grpc异常
            return None

        others_infos_protobuf = response.others_infos

        others_infos = protobuf_to_dict(others_infos_protobuf) # TODO: 没有判断comm_mask字段是否存在

        return others_infos

    def get_others_comm_masks(self):  # 从协同感知子系统获取所有他车协作图
        if self.cfg.rpc_collaboration_client_debug:
            return {'DEBUG': {'comm_mask': np.ones((1, 3)),
                              'ts_comm_mask': 1}}

        try:
            response = self.__collaboration_stub.GetOthersCommMasks(Service_pb2.Empty(), timeout=10)  # 请求协同感知子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC get_others_comm_masks failed: code={e.code().name}")  # 记录grpc异常
            return None

        others_comm_masks_protobuf = response.others_comm_masks

        others_comm_masks = protobuf_to_dict(others_comm_masks_protobuf)

        return others_comm_masks

    def get_others_lidar_poses_and_pcds(self):
        if self.cfg.rpc_collaboration_client_debug:
            return {'DEBUG': {'lidar_pose': np.ones((1, 3)),
                              'ts_lidar_pose': 1,
                              'pcd': np.ones((1, 3)),
                              'ts_pcd': 1}}

        try:
            response = self.__collaboration_stub.GetOthersLidarPosesAndPCDs(Service_pb2.Empty(), timeout=10)  # 请求协同感知子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC get_others_lidar_poses_and_pcds failed: code={e.code().name}")  # 记录grpc异常
            return None

        others_lidar_poses_and_pcds_protobuf = response.others_lidar_poses_and_pcds

        others_lidar_poses_and_pcds = protobuf_to_dict(others_lidar_poses_and_pcds_protobuf)

        return others_lidar_poses_and_pcds
