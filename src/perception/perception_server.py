import threading

import grpc
from concurrent import futures
from . import Service_pb2
from . import Service_pb2_grpc
import time
import numpy as np


class MyInfo:
    def __init__(self):
        self.pcd = np.array([])  # 初始化为空数组
        self.feature = {}  # 初始化为空字典
        self.conf_map = np.array([])  # 初始化为空数组
        self.comm_mask = np.array([])  # 初始化为空数组
        self.pose = np.array([])  # 初始化为空数组
        self.velocity = np.array([])  # 初始化为空数组
        self.acceleration = np.array([])  # 初始化为空数组
        self.extrinsic_matrix = np.array([])  # 初始化为空数组

        self.pcd_lock = threading.Lock()
        self.feature_lock = threading.Lock()
        self.conf_map_lock = threading.Lock()
        self.comm_mask_lock = threading.Lock()
        self.pose_lock = threading.Lock()
        self.velocity_lock = threading.Lock()
        self.acceleration_lock = threading.Lock()
        self.extrinsic_matrix_lock = threading.Lock()

    def update_pcd(self, pcd):
        with self.pcd_lock:
            self.pcd = pcd  # 线程安全更新

    def update_feature(self, feature):
        with self.feature_lock:
            self.feature = feature  # 线程安全更新

    def update_conf_map(self, conf_map):
        with self.conf_map_lock:
            self.conf_map = conf_map  # 线程安全更新

    def update_comm_mask(self, comm_mask):
        with self.comm_mask_lock:
            self.comm_mask = comm_mask  # 线程安全更新

    def update_pose(self, pose):
        with self.pose_lock:
            self.pose = pose  # 线程安全更新

    def update_velocity(self, velocity):
        with self.velocity_lock:
            self.velocity = velocity  # 线程安全更新

    def update_acceleration(self, acceleration):
        with self.acceleration_lock:
            self.acceleration = acceleration  # 线程安全更新

    def update_extrinsic_matrix(self, extrinsic_matrix):
        with self.extrinsic_matrix_lock:
            self.extrinsic_matrix = extrinsic_matrix  # 线程安全更新

    def get_pcd_copy(self):
        with self.pcd_lock:
            return self.pcd.copy() if self.pcd.size > 0 else self.pcd

    def get_feature_copy(self):
        with self.feature_lock:
            return self.feature

    def get_conf_map_copy(self):
        with self.conf_map_lock:
            return self.conf_map.copy() if self.conf_map.size > 0 else self.conf_map

    def get_comm_mask_copy(self):
        with self.comm_mask_lock:
            return self.comm_mask.copy() if self.comm_mask.size > 0 else self.comm_mask

    def get_pose_copy(self):
        with self.pose_lock:
            return self.pose.copy() if self.pose.size > 0 else self.pose

    def get_velocity_copy(self):
        with self.velocity_lock:
            return self.velocity.copy() if self.velocity.size > 0 else self.velocity

    def get_acceleration_copy(self):
        with self.acceleration_lock:
            return self.acceleration.copy() if self.acceleration.size > 0 else self.acceleration

    def get_extrinsic_matrix_copy(self):
        with self.extrinsic_matrix_lock:
            return self.extrinsic_matrix.copy() if self.extrinsic_matrix.size > 0 else self.extrinsic_matrix


class PerceptionService(Service_pb2_grpc.PerceptionServiceServicer):  # 感知子系统的Service类
    def __init__(self, my_info):
        self.my_info = my_info

    def GetMyPCD(self, request, context):           # 感知子系统向其他进程提供“获取自车点云”的服务
        timestamp = int(time.time())        # 时间戳
        my_pcd = self.my_info.get_pcd_copy()

        return Service_pb2.PCD(  # 序列化并返回自车点云
            timestamp=timestamp,
            pcd=Service_pb2.NdArray(
                data=my_pcd.tobytes(),
                dtype=str(my_pcd.dtype),
                shape=list(my_pcd.shape)
            )
        )

    def GetMyFeature(self, request, context):  # 感知子系统向其他进程提供“获取自车特征”的服务
        timestamp = int(time.time())                        # 时间戳
        my_feature = self.my_info.get_feature_copy()

        return Service_pb2.Feature(  # 序列化并返回自车特征
            timestamp=timestamp,
            feature=Service_pb2._Feature(
                voxel_features=Service_pb2.NdArray(
                    data=my_feature['voxel_features'].tobytes(),
                    dtype=str(my_feature['voxel_features'].dtype),
                    shape=list(my_feature['voxel_features'].shape)
                ),
                voxel_coords=Service_pb2.NdArray(
                    data=my_feature['voxel_coords'].tobytes(),
                    dtype=str(my_feature['voxel_coords'].dtype),
                    shape=list(my_feature['voxel_coords'].shape)
                ),
                voxel_num_points=Service_pb2.NdArray(
                    data=my_feature['voxel_num_points'].tobytes(),
                    dtype=str(my_feature['voxel_num_points'].dtype),
                    shape=list(my_feature['voxel_num_points'].shape)
                )
            )
        )

    def GetMyConfMap(self, request, context):  # 感知子系统向其他进程提供“获取自车置信图”的服务
        timestamp = int(time.time())        # 时间戳
        my_conf_map = self.my_info.get_conf_map_copy()

        return Service_pb2.ConfMap(  # 序列化并返回自车置信图
            timestamp=timestamp,
            conf_map=Service_pb2.NdArray(
                data=my_conf_map.tobytes(),
                dtype=str(my_conf_map.dtype),
                shape=list(my_conf_map.shape)
            )
        )

    def GetMyCommMask(self, request, context):  # 感知子系统向其他进程提供“获取自车协作图”的服务
        timestamp = int(time.time())        # 时间戳
        my_comm_mask = self.my_info.get_comm_mask_copy()

        return Service_pb2.CommMask(  # 序列化并返回自车协作图
            timestamp=timestamp,
            comm_mask=Service_pb2.NdArray(
                data=my_comm_mask.tobytes(),
                dtype=str(my_comm_mask.dtype),
                shape=list(my_comm_mask.shape)
            )
        )

    def GetMyPVAInfo(self, request, context):   # 感知子系统向其他进程提供“获取自车位置、速度、加速度信息”的服务
        timestamp = int(time.time())            # 时间戳
        pose = self.my_info.get_pose_copy()
        velocity = self.my_info.get_velocity_copy()
        acceleration = self.my_info.get_acceleration_copy()

        return Service_pb2.PVAInfo(  # 序列化并返回自车位置、速度、加速度信息
            timestamp=timestamp,
            pose=Service_pb2.NdArray(
                data=pose.tobytes(),
                dtype=str(pose.dtype),
                shape=list(pose.shape)
            ),
            velocity=Service_pb2.NdArray(
                data=velocity.tobytes(),
                dtype=str(velocity.dtype),
                shape=list(velocity.shape)
            ),
            acceleration=Service_pb2.NdArray(
                data=acceleration.tobytes(),
                dtype=str(acceleration.dtype),
                shape=list(acceleration.shape)
            )
        )

    def GetMyExtrinsicMatrix(self, request, context):   # 感知子系统向其他进程提供“获取自车外参矩阵”的服务
        timestamp = int(time.time())                    # 时间戳
        my_extrinsic_matrix = self.my_info.get_extrinsic_matrix_copy()

        return Service_pb2.ExtrinsicMatrix(  # 序列化并返回自车外参矩阵
            timestamp=timestamp,
            extrinsic_matrix=Service_pb2.NdArray(
                data=my_extrinsic_matrix.tobytes(),
                dtype=str(my_extrinsic_matrix.dtype),
                shape=list(my_extrinsic_matrix.shape)
            )
        )


class PerceptionServerThread(threading.Thread):                                 # 感知子系统的Server线程
    def __init__(self, my_pcd):
        super().__init__()
        self.my_pcd = my_pcd

    def run(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
            ('grpc.max_send_message_length', 64 * 1024 * 1024),                 # 设置gRPC 消息的最大发送和接收大小为64MB
            ('grpc.max_receive_message_length', 64 * 1024 * 1024)])
        Service_pb2_grpc.add_PerceptionServiceServicer_to_server(PerceptionService(self.my_pcd), server)
        server.add_insecure_port('[::]:50051')
        server.start()                              # 非阻塞, 会实例化一个新线程来处理请求
        print("Perception Server is up and running on port 50051.")
        try:
            server.wait_for_termination()           # 保持服务器运行直到终止
        except KeyboardInterrupt:
            server.stop(0)                          # 服务器终止
            print("Perception Server terminated.")
