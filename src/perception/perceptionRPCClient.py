import logging
from appConfig import AppConfig
import grpc
import numpy as np

from rpc import Service_pb2
from rpc import Service_pb2_grpc


class PerceptionRPCClient:                                 # 感知子系统的Client类，用于向感知子系统的服务器请求服务
    def __init__(self, cfg: AppConfig):
        perception_channel = grpc.insecure_channel('localhost:50051', options=[                 # 与感知子系统建立连接
            ('grpc.max_send_message_length', 64 * 1024 * 1024),                     # 设置gRPC 消息的最大发送和接收大小为64MB
            ('grpc.max_receive_message_length', 64 * 1024 * 1024)])
        self.__perception_stub = Service_pb2_grpc.PerceptionServiceStub(perception_channel)
        self.cfg = cfg

    def get_my_pcd(self):       # 从感知子系统获取自车点云
        if self.cfg.rpc_perception_client_debug:
            return 0, np.ones((1, 4))

        try:
            response = self.__perception_stub.GetMyPCD(Service_pb2.Empty(), timeout=5)  # 请求感知子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code()}")  # 记录grpc异常
            return None, None

        timestamp = response.timestamp  # 时间戳
        # 自车点云
        my_pcd = np.frombuffer(response.pcd.data,
                               dtype=response.pcd.dtype).reshape(response.pcd.shape)
        return timestamp, my_pcd

    def get_my_pose_and_pcd(self):       # 从感知子系统获取自车雷达位姿和点云
        if self.cfg.rpc_perception_client_debug:
            return 0, np.array((6, )), np.array((1, 4))

        try:
            response = self.__perception_stub.GetMyPoseAndPCD(Service_pb2.Empty(), timeout=5)  # 请求感知子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code()}")  # 记录grpc异常
            return None, None, None

        timestamp = response.timestamp  # 时间戳
        # 自车雷达位姿
        my_pose = np.frombuffer(response.pose.data,
                                dtype=response.pose.dtype).reshape(response.pose.shape)
        # 自车点云
        my_pcd = np.frombuffer(response.pcd.data,
                               dtype=response.pcd.dtype).reshape(response.pcd.shape)
        return timestamp, my_pose, my_pcd

    def get_my_feature(self):  # 从感知子系统获取自车特征
        if self.cfg.rpc_perception_client_debug:
            return 0, {'voxel_features': np.ones((1, 1)), 'voxel_coords': np.ones((1, 1)), 'voxel_num_points': np.ones((1, 1))}

        try:
            response = self.__perception_stub.GetMyFeature(Service_pb2.Empty(), timeout=5)  # 请求感知子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code()}")  # 记录grpc异常
            return None, None

        timestamp = response.timestamp  # 时间戳
        # 体素特征
        voxel_features_message = response.feature.voxel_features
        voxel_features = np.frombuffer(voxel_features_message.data,
                                       dtype=voxel_features_message.dtype).reshape(voxel_features_message.shape)
        # 体素坐标
        voxel_coords_message = response.feature.voxel_coords
        voxel_coords = np.frombuffer(voxel_coords_message.data,
                                     dtype=voxel_coords_message.dtype).reshape(voxel_coords_message.shape)

        # 体素点数
        voxel_num_points_message = response.feature.voxel_num_points
        voxel_num_points = np.frombuffer(voxel_num_points_message.data,
                                         dtype=voxel_num_points_message.dtype).reshape(voxel_num_points_message.shape)
        # 自车特征
        my_feature = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points}
        return timestamp, my_feature

    def get_my_conf_map(self):  # 从感知子系统获取自车置信图
        if self.cfg.rpc_perception_client_debug:
            return 0, np.ones((1, 1))

        try:
            response = self.__perception_stub.GetMyConfMap(Service_pb2.Empty(), timeout=5)  # 请求感知子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code()}")  # 记录grpc异常
            return None, None

        timestamp = response.timestamp  # 时间戳
        # 自车置信图
        my_conf_map = np.frombuffer(response.conf_map.data,
                                    dtype=response.conf_map.dtype).reshape(response.conf_map.shape)
        return timestamp, my_conf_map

    def get_my_comm_mask(self):  # 从感知子系统获取自车协作图
        if self.cfg.rpc_perception_client_debug:
            return 0, np.ones((1, 1))

        try:
            response = self.__perception_stub.GetMyCommMask(Service_pb2.Empty(), timeout=5)  # 请求感知子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code()}")  # 记录grpc异常
            return None, None

        timestamp = response.timestamp  # 时间戳
        # 自车协作图
        my_comm_mask = np.frombuffer(response.comm_mask.data,
                                     dtype=response.comm_mask.dtype).reshape(response.comm_mask.shape)
        return timestamp, my_comm_mask

    def get_my_pva_info(self):  # 从感知子系统获取自车位置、速度、加速度信息
        if self.cfg.rpc_perception_client_debug:
            return 0, np.ones((1, 6)), np.ones((1, )), np.ones((1, ))

        try:
            response = self.__perception_stub.GetMyPVAInfo(Service_pb2.Empty(), timeout=5)  # 请求感知子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code()}")  # 记录grpc异常
            return None, None, None, None

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

    def get_my_extrinsic_matrix(self):  # 从感知子系统获取自车外参矩阵
        if self.cfg.rpc_perception_client_debug:
            return 0, np.ones((4, 4))

        try:
            response = self.__perception_stub.GetMyExtrinsicMatrix(Service_pb2.Empty(), timeout=5)  # 请求感知子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code()}")  # 记录grpc异常
            return None, None

        timestamp = response.timestamp  # 时间戳
        # 自车外参矩阵
        my_extrinsic_matrix = np.frombuffer(response.extrinsic_matrix.data,
                                            dtype=response.extrinsic_matrix.dtype).reshape(response.extrinsic_matrix.shape)
        return timestamp, my_extrinsic_matrix
