import logging
import threading

import grpc
from concurrent import futures

import numpy as np

from rpc import Service_pb2
from rpc import Service_pb2_grpc
import time


class CollaborationService(Service_pb2_grpc.CollaborationServiceServicer):  # 协同感知子系统的Service类
    def GetOthersInfo(self, request, context):  # 协同感知子系统向其他进程提供“获取所有他车信息”的服务
        ids = []                # 所有他车的id
        timestamps = []         # 所有他车传递信息对应的时间戳
        poses = []              # 所有他车的位置
        velocities = []         # 所有他车的速度
        accelerations = []      # 所有他车的加速度
        voxel_features = []     # 所有他车体素特征
        voxel_coords = []       # 所有他车体素坐标
        voxel_num_points = []   # 所有他车体素点数

        # ###################################################需要真实数据来源
        for i in range(5):
            ids.append(i)
            timestamps.append(int(time.time()))

            poses.append(i * 100 + np.array([1, 2, 3]))
            velocities.append(i * 100 + np.array([4, 5, 6]))
            accelerations.append(i * 100 + np.array([7, 8, 9]))
            voxel_features.append(i * 100 + np.array([10, 11, 12]))
            voxel_coords.append(i * 100 + np.array([13, 14, 15]))
            voxel_num_points.append(i * 100 + np.array([16, 17, 18]))

            time.sleep(1)
        # ###################################################

        poses = np.stack(poses, axis=0)  # 合并所有位置信息
        velocities = np.stack(velocities, axis=0)  # 合并所有速度信息
        accelerations = np.stack(accelerations, axis=0)  # 合并所有加速度信息
        voxel_features = np.stack(voxel_features, axis=0)  # 合并所有体素特征
        voxel_coords = np.stack(voxel_coords, axis=0)  # 合并所有体素坐标
        voxel_num_points = np.stack(voxel_num_points, axis=0)  # 合并所有体素点数
        return Service_pb2.OthersInfo(  # 序列化并返回所有车辆信息
            ids=ids,
            timestamps=timestamps,
            poses=Service_pb2.NdArray(
                data=poses.tobytes(),
                dtype=str(poses.dtype),
                shape=list(poses.shape)
            ),
            velocities=Service_pb2.NdArray(
                data=velocities.tobytes(),
                dtype=str(velocities.dtype),
                shape=list(velocities.shape)
            ),
            accelerations=Service_pb2.NdArray(
                data=accelerations.tobytes(),
                dtype=str(accelerations.dtype),
                shape=list(accelerations.shape)
            ),
            voxel_features=Service_pb2.NdArray(
                data=voxel_features.tobytes(),
                dtype=str(voxel_features.dtype),
                shape=list(voxel_features.shape)
            ),
            voxel_coords=Service_pb2.NdArray(
                data=voxel_coords.tobytes(),
                dtype=str(voxel_coords.dtype),
                shape=list(voxel_coords.shape)
            ),
            voxel_num_points=Service_pb2.NdArray(
                data=voxel_num_points.tobytes(),
                dtype=str(voxel_num_points.dtype),
                shape=list(voxel_num_points.shape)
            )
        )

    def GetOthersCommMasks(self, request, context):  # 协同感知子系统向其他进程提供“获取所有他车协作图”的服务
        ids = []  # 所有他车的id
        timestamps = []  # 所有他车传递协作图对应时间戳
        others_comm_masks = []  # 所有他车的协作图

        # ###################################################需要真实数据来源
        for i in range(5):
            ids.append(i)
            timestamps.append(int(time.time()))

            others_comm_masks.append(i * 10 + np.array([1, 2, 3]))

            time.sleep(1)
        # ###################################################

        others_comm_masks = np.stack(others_comm_masks, axis=0)  # 合并所有协作图

        return Service_pb2.OthersCommMasks(  # 序列化并返回所有他车协作图
            ids=ids,
            timestamps=timestamps,
            others_comm_masks=Service_pb2.NdArray(
                data=others_comm_masks.tobytes(),
                dtype=str(others_comm_masks.dtype),
                shape=list(others_comm_masks.shape)
            )
        )


class CollaborationServerThread(threading.Thread):
    def run(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
            ('grpc.max_send_message_length', 64 * 1024 * 1024),                     # 设置gRPC 消息的最大发送和接收大小为64MB
            ('grpc.max_receive_message_length', 64 * 1024 * 1024)])
        Service_pb2_grpc.add_CollaborationServiceServicer_to_server(CollaborationService(), server)
        server.add_insecure_port('[::]:50052')
        server.start()                              # 非阻塞, 会实例化一个新线程来处理请求
        print("Collaboration Server is up and running on port 50052.")
        try:
            server.wait_for_termination()           # 保持服务器运行直到终止
        except KeyboardInterrupt:
            server.stop(0)                          # 服务器终止
            print("Collaboration Server terminated.")


class CollaborationClient:          # 协同感知子系统的Client类，用于请求其他进程的服务
    def __init__(self):
        perception_channel = grpc.insecure_channel('localhost:50051', options=[                # 与感知子系统建立连接
            ('grpc.max_send_message_length', 64 * 1024 * 1024),                     # 设置gRPC 消息的最大发送和接收大小为64MB
            ('grpc.max_receive_message_length', 64 * 1024 * 1024)])
        self.__perception_stub = Service_pb2_grpc.PerceptionServiceStub(perception_channel)

    def get_my_feature(self):   # 从感知子系统获取自车特征
        try:
            response = self.__perception_stub.GetMyFeature(Service_pb2.Empty(), timeout=5)     # 请求感知子系统并获得响应
        except grpc.RpcError as e:                                                  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code}, details={e.details}")        # 记录grpc异常
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
            response = self.__perception_stub.GetMyConfMap(Service_pb2.Empty(), timeout=5)       # 请求感知子系统并获得响应
        except grpc.RpcError as e:                                                  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code}, details={e.details}")        # 记录grpc异常
            return -1, -1

        timestamp = response.timestamp  # 时间戳
        # 自车置信图
        my_conf_map = np.frombuffer(response.my_conf_map.data,
                                    dtype=response.my_conf_map.dtype).reshape(response.my_conf_map.shape)
        return timestamp, my_conf_map

    def get_my_comm_mask(self):  # 从感知子系统获取自车协作图
        try:
            response = self.__perception_stub.GetMyCommMask(Service_pb2.Empty(), timeout=5)    # 请求感知子系统并获得响应
        except grpc.RpcError as e:                                                  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code}, details={e.details}")        # 记录grpc异常
            return -1, -1

        timestamp = response.timestamp  # 时间戳
        # 自车协作图
        my_comm_mask = np.frombuffer(response.my_comm_mask.data,
                                     dtype=response.my_comm_mask.dtype).reshape(response.my_comm_mask.shape)
        return timestamp, my_comm_mask

    def get_my_pva_info(self):      # 从感知子系统获取自车位置、速度、加速度信息
        try:
            response = self.__perception_stub.GetMyPVAInfo(Service_pb2.Empty(), timeout=5)     # 请求感知子系统并获得响应
        except grpc.RpcError as e:                                                  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code}, details={e.details}")        # 记录grpc异常
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

    def get_my_extrinsic_matrix(self):      # 从感知子系统获取自车外参矩阵
        try:
            response = self.__perception_stub.GetMyExtrinsicMatrix(Service_pb2.Empty(), timeout=5)  # 请求感知子系统并获得响应
        except grpc.RpcError as e:                                                  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code}, details={e.details}")        # 记录grpc异常
            return -1, -1

        timestamp = response.timestamp  # 时间戳
        # 自车外参矩阵
        my_extrinsic_matrix = np.frombuffer(response.my_extrinsic_matrix.data,
                                            dtype=response.my_extrinsic_matrix.dtype).reshape(response.my_extrinsic_matrix.shape)
        return timestamp, my_extrinsic_matrix
