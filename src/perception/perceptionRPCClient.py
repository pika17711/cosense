import logging
from appConfig import AppConfig
import grpc
import numpy as np

from rpc import Service_pb2
from rpc import Service_pb2_grpc
from utils.rpc_utils import protobuf_to_np


class PerceptionRPCClient:                                 # 感知子系统的Client类，用于向感知子系统的服务器请求服务
    def __init__(self, cfg: AppConfig):
        perception_channel = grpc.insecure_channel('localhost:50051', options=[                 # 与感知子系统建立连接
            ('grpc.max_send_message_length', 64 * 1024 * 1024),                     # 设置gRPC 消息的最大发送和接收大小为64MB
            ('grpc.max_receive_message_length', 64 * 1024 * 1024)])
        self.__perception_stub = Service_pb2_grpc.PerceptionServiceStub(perception_channel)
        self.cfg = cfg

    def get_my_pcd(self):       # 从感知子系统获取自车点云
        if self.cfg.rpc_perception_client_debug:
            return np.ones((1, 4)), 0

        try:
            response = self.__perception_stub.GetMyPCD(Service_pb2.Empty(), timeout=5)  # 请求感知子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC get_my_pcd failed: code={e.code()}")  # 记录grpc异常
            return None, None

        # 自车点云
        my_pcd = protobuf_to_np(response.pcd)
        ts_pcd = response.ts_pcd  # 时间戳

        return my_pcd, ts_pcd

    def get_my_lidar_pose_and_pcd(self):       # 从感知子系统获取自车雷达位姿和点云
        if self.cfg.rpc_perception_client_debug:
            return np.array((6, )), 0, np.array((1, 4)), 0

        try:
            response = self.__perception_stub.GetMyLidarPoseAndPCD(Service_pb2.Empty(), timeout=5)  # 请求感知子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC get_my_lidar_pose_and_pcd failed: code={e.code()}")  # 记录grpc异常
            return None, None, None, None

        # 自车雷达位姿
        my_lidar_pose = protobuf_to_np(response.lidar_pose)
        ts_lidar_pose = response.ts_lidar_pose
        # 自车点云
        my_pcd = protobuf_to_np(response.pcd).copy()
        ts_pcd = response.ts_pcd
        return my_lidar_pose, ts_lidar_pose, my_pcd, ts_pcd

    def get_my_pva(self):  # 从感知子系统获取自车位置、速度、加速度信息
        if self.cfg.rpc_perception_client_debug:
            return np.ones((1, 6)), 0, np.ones((1, )), 0, np.ones((1, )), 0

        try:
            response = self.__perception_stub.GetMyPVA(Service_pb2.Empty(), timeout=5)  # 请求感知子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC get_my_pva failed: code={e.code()}")  # 记录grpc异常
            return None, None, None, None, None, None

        # 自车的位置
        lidar_pose = protobuf_to_np(response.lidar_pose)
        ts_lidar_pose = response.ts_lidar_pose
        # 自车的速度
        speed = protobuf_to_np(response.speed)
        ts_spd = response.ts_spd
        # 自车的加速度
        acceleration = protobuf_to_np(response.acceleration)
        ts_acc = response.ts_acc
        return lidar_pose, ts_lidar_pose, speed, ts_spd, acceleration, ts_acc

    def get_perception_info(self):
        if self.cfg.rpc_perception_client_debug:
            return np.ones((1, 6)), 0, np.ones((1, )), 0, np.ones((1, )), 0, np.ones((1, 3)), 0

        try:
            response = self.__perception_stub.GetPerceptionInfo(Service_pb2.Empty(), timeout=5)  # 请求感知子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC get_perception_info failed: code={e.code()}")  # 记录grpc异常
            return None, None, None, None, None, None, None, None

        # 自车的位置
        lidar_pose = protobuf_to_np(response.lidar_pose)
        ts_lidar_pose = response.ts_lidar_pose
        # 自车的速度
        speed = protobuf_to_np(response.speed)
        ts_spd = response.ts_spd
        # 自车的加速度
        acceleration = protobuf_to_np(response.acceleration)
        ts_acc = response.ts_acc
        # 自车点云
        my_pcd = protobuf_to_np(response.pcd).copy()
        ts_pcd = response.ts_pcd  # 时间戳

        return lidar_pose, ts_lidar_pose, speed, ts_spd, acceleration, ts_acc, my_pcd, ts_pcd

    def get_my_extrinsic_matrix(self):  # 从感知子系统获取自车外参矩阵
        if self.cfg.rpc_perception_client_debug:
            return np.ones((4, 4)), 0

        try:
            response = self.__perception_stub.GetMyExtrinsicMatrix(Service_pb2.Empty(), timeout=5)  # 请求感知子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC get_my_extrinsic_matrix failed: code={e.code()}")  # 记录grpc异常
            return None, None

        # 自车外参矩阵
        my_extrinsic_matrix = protobuf_to_np(response.extrinsic_matrix)
        ts_extrinsic_matrix = response.ts_extrinsic_matrix  # 时间戳
        return my_extrinsic_matrix, ts_extrinsic_matrix
