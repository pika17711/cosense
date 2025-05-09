import logging
from typing import Dict, Tuple
from config import AppConfig
import grpc
import numpy as np

from rpc import Service_pb2
from rpc import Service_pb2_grpc
from utils.common import mstime


class PerceptionClient:                                 # 感知子系统的Client类，用于向感知子系统的服务器请求服务
    def __init__(self):
        perception_channel = grpc.insecure_channel('localhost:50051', options=[                 # 与感知子系统建立连接
            ('grpc.max_send_message_length', 64 * 1024 * 1024),                     # 设置gRPC 消息的最大发送和接收大小为64MB
            ('grpc.max_receive_message_length', 64 * 1024 * 1024)])
        self.__perception_stub = Service_pb2_grpc.PerceptionServiceStub(perception_channel)

    def get_my_feature(self) -> Tuple[AppConfig.timestamp_t, Dict[str, np.ndarray]]:  # 从感知子系统获取自车特征
        return mstime(), {'voxel_features': np.array([1]), 'voxel_coords': np.array([1]), 'voxel_num_points': np.array(1)}

        try:
            response = self.__perception_stub.GetMyFeature(Service_pb2.Empty(), timeout=5)  # 请求感知子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code}, details={e.details}")  # 记录grpc异常
            return -1, -1

        timestamp = response.timestamp  # 时间戳
        # 体素特征
        voxel_features_message = response.my_feature.voxel_features
        voxel_features = np.frombuffer(voxel_features_message.data,
                                       dtype=voxel_features_message.dtype).reshape(voxel_features_message.shape)
        # 体素坐标
        voxel_coords_message = response.my_feature.voxel_coords
        voxel_coords = np.frombuffer(voxel_coords_message.data,
                                     dtype=voxel_coords_message.dtype).reshape(voxel_coords_message.shape)

        # 体素点数
        voxel_num_points_message = response.my_feature.voxel_num_points
        voxel_num_points = np.frombuffer(voxel_num_points_message.data,
                                         dtype=voxel_num_points_message.dtype).reshape(voxel_num_points_message.shape)
        # 自车特征
        my_feature = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points}
        return timestamp, my_feature

    def get_my_conf_map(self):  # 从感知子系统获取自车置信图
        try:
            response = self.__perception_stub.GetMyConfMap(Service_pb2.Empty(), timeout=5)  # 请求感知子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code}, details={e.details}")  # 记录grpc异常
            return -1, -1

        timestamp = response.timestamp  # 时间戳
        # 自车置信图
        my_conf_map = np.frombuffer(response.my_conf_map.data,
                                    dtype=response.my_conf_map.dtype).reshape(response.my_conf_map.shape)
        return timestamp, my_conf_map

    def get_my_comm_mask(self):  # 从感知子系统获取自车协作图
        try:
            response = self.__perception_stub.GetMyCommMask(Service_pb2.Empty(), timeout=5)  # 请求感知子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code}, details={e.details}")  # 记录grpc异常
            return -1, -1

        timestamp = response.timestamp  # 时间戳
        # 自车协作图
        my_comm_mask = np.frombuffer(response.my_comm_mask.data,
                                     dtype=response.my_comm_mask.dtype).reshape(response.my_comm_mask.shape)
        return timestamp, my_comm_mask

    def get_my_pva_info(self) -> Tuple[AppConfig.timestamp_t, np.ndarray, np.ndarray, np.ndarray]:  # 从感知子系统获取自车位置、速度、加速度信息
        return mstime(), np.array([1, 2, 3]), np.array([1]), np.array([1])

        try:
            response = self.__perception_stub.GetMyPVAInfo(Service_pb2.Empty(), timeout=5)  # 请求感知子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code}, details={e.details}")  # 记录grpc异常
            return -1, -1, -1, -1

        timestamp = response.timestamp  # 时间戳
        # 自车的位置
        pose = np.frombuffer(response.pose.data,
                             dtype=response.pose.dtype).reshape(response.pose.shape)
        # 自车的速度
        velocity = np.frombuffer(response.velocity.data,
                                 dtype=response.velocity.dtype).reshape(response.velocity.shape)
        # 自车的加速度
        acceleration = np.frombuffer(response.acceleration.data,
                                     dtype=response.acceleration.dtype).reshape(response.acceleration.shape)
        return timestamp, pose, velocity, acceleration

    def get_my_extrinsic_matrix(self) -> Tuple[AppConfig.timestamp_t, np.ndarray]:  # 从感知子系统获取自车外参矩阵
        return mstime(), np.eye(4)

        try:
            response = self.__perception_stub.GetMyExtrinsicMatrix(Service_pb2.Empty(), timeout=5)  # 请求感知子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code}, details={e.details}")  # 记录grpc异常
            return -1, -1

        timestamp = response.timestamp  # 时间戳
        # 自车外参矩阵
        my_extrinsic_matrix = np.frombuffer(response.my_extrinsic_matrix.data,
                                            dtype=response.my_extrinsic_matrix.dtype).reshape(
            response.my_extrinsic_matrix.shape)
        return timestamp, my_extrinsic_matrix
