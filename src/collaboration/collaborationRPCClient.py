import logging
import grpc
import numpy as np

from rpc import Service_pb2
from rpc import Service_pb2_grpc

from appConfig import AppConfig

class CollaborationRPCClient:          # 协同感知子系统的Client类，用于向协同感知子系统的服务器请求服务
    def __init__(self, cfg: AppConfig):
        collaboration_channel = grpc.insecure_channel('localhost:50052', options=[          # 与协同感知子系统的服务器建立连接
            ('grpc.max_send_message_length', 64 * 1024 * 1024),                     # 设置gRPC 消息的最大发送和接收大小为64MB
            ('grpc.max_receive_message_length', 64 * 1024 * 1024)])
        self.__collaboration_stub = Service_pb2_grpc.CollaborationServiceStub(collaboration_channel)
        self.cfg = cfg

    def get_others_poses_and_pcds(self):  # 从协同感知子系统获取所有他车雷达位姿和点云
        if self.cfg.rpc_collaboration_client_debug:
            return ['DEBUG'], [0], np.ones((1, 3)), np.ones((1, 1, 4))

        try:
            response = self.__collaboration_stub.GetOthersPosesAndPCDs(Service_pb2.Empty(), timeout=10)  # 请求协同感知子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code().name}")  # 记录grpc异常
            return None, None, None, None

        ids = response.ids  # 所有他车的id
        timestamps = response.timestamps  # 所有他车传递协作图对应的时间戳
        # 所有他车的雷达位姿
        others_poses = np.frombuffer(response.poses.data, dtype=response.poses.dtype).reshape(response.poses.shape)
        # 所有他车的点云
        others_pcds = np.frombuffer(response.PCDs.data, dtype=response.PCDs.dtype).reshape(response.PCDs.shape)
        return ids, timestamps, others_poses, others_pcds

    def get_others_info(self):  # 从协同感知子系统获取所有他车信息
        if self.cfg.rpc_collaboration_client_debug:
            return ['DEBUG'], [1], np.ones((1, 3)), np.ones((1, 3)), np.ones((1, 3)), 1, np.ones((1, 3)), np.ones((1, 3)), np.ones((1, 3))
        
        try:
            response = self.__collaboration_stub.GetOthersInfo(Service_pb2.Empty(), timeout=10)  # 请求协同感知子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code().name}")  # 记录grpc异常
            return None, None, None, None, None, None, None, None, None

        ids = response.ids  # 所有他车的id
        timestamps = response.timestamps  # 所有他车传递信息对应的时间戳
        # 所有他车的位置
        poses = np.frombuffer(response.poses.data,
                              dtype=response.poses.dtype).reshape(response.poses.shape)
        # 所有他车的速度
        velocities = np.frombuffer(response.velocities.data,
                                   dtype=response.velocities.dtype).reshape(response.velocities.shape)
        # 所有他车的加速度
        accelerations = np.frombuffer(response.accelerations.data,
                                      dtype=response.accelerations.dtype).reshape(response.accelerations.shape)
        features_lens = response.features_lens
        # 所有他车的体素特征
        voxel_features = np.frombuffer(response.voxel_features.data,
                                       dtype=response.voxel_features.dtype).reshape(response.voxel_features.shape)
        # 所有他车的体素坐标
        voxel_coords = np.frombuffer(response.voxel_coords.data,
                                     dtype=response.voxel_coords.dtype).reshape(response.voxel_coords.shape)
        # 所有他车的体素点数
        voxel_num_points = np.frombuffer(response.voxel_num_points.data,
                                         dtype=response.voxel_num_points.dtype).reshape(response.voxel_num_points.shape)

        return ids, timestamps, poses, velocities, accelerations, \
            features_lens, voxel_features, voxel_coords, voxel_num_points

    def get_others_comm_masks(self):  # 从协同感知子系统获取所有他车协作图
        if self.cfg.rpc_collaboration_client_debug:
            return ['DEBUG'], [1], np.ones((1, 3, 3))

        try:
            response = self.__collaboration_stub.GetOthersCommMasks(Service_pb2.Empty(), timeout=10)  # 请求协同感知子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code().name}")  # 记录grpc异常
            return None, None, None

        ids = response.ids  # 所有他车的id
        timestamps = response.timestamps  # 所有他车传递协作图对应的时间戳
        # 所有他车的协作图
        others_comm_masks = np.frombuffer(response.others_comm_masks.data,
                                          dtype=response.others_comm_masks.dtype).reshape(
            response.others_comm_masks.shape)
        
        return ids, timestamps, others_comm_masks
