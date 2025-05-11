import threading
import grpc
import time
import numpy as np
from concurrent import futures

from rpc import Service_pb2
from rpc import Service_pb2_grpc


class SharedOthersInfo:
    def __init__(self):
        self.__ids = []               # 初始化为空数组
        self.__timestamps = []        # 初始化为空数组
        self.__poses = np.array([])   # 初始化为空数组
        self.__pcds = np.array([])    # 初始化为空数组

        self.__ids_lock = threading.Lock()
        self.__timestamps_lock = threading.Lock()
        self.__poses_lock = threading.Lock()
        self.__pcds_lock = threading.Lock()

    def update_ids(self, ids):
        with self.__ids_lock:
            self.__ids = ids  # 线程安全更新

    def update_timestamps(self, timestamps):
        with self.__timestamps_lock:
            self.__timestamps = timestamps  # 线程安全更新

    def update_poses(self, poses):
        with self.__poses_lock:
            self.__poses = poses  # 线程安全更新

    def update_pcds(self, pcds):
        with self.__pcds_lock:
            self.__pcds = pcds  # 线程安全更新

    def get_ids_copy(self):
        with self.__ids_lock:
            return self.__ids.copy()

    def get_timestamps_copy(self):
        with self.__timestamps_lock:
            return self.__timestamps.copy()

    def get_poses_copy(self):
        with self.__poses_lock:
            return self.__poses.copy() if self.__poses.size > 0 else self.__poses

    def get_pcds_copy(self):
        with self.__pcds_lock:
            return self.__pcds.copy() if self.__pcds.size > 0 else self.__pcds


class CollaborationService(Service_pb2_grpc.CollaborationServiceServicer):  # 协同感知子系统的Service类
    def __init__(self, others_info):
        self.others_info = others_info

    def GetOthersPosesAndPCDs(self, request, context):  # 协同感知子系统向其他进程提供“获取所有他车雷达位姿和点云”的服务
        ids = self.others_info.get_ids_copy()
        timestamps = self.others_info.get_timestamps_copy()
        poses = self.others_info.get_poses_copy()
        pcds = self.others_info.get_pcds_copy()

        return Service_pb2.OthersPosesAndPCDs(  # 序列化并返回所有他车雷达位姿和点云
            ids=ids,
            timestamps=timestamps,
            poses=Service_pb2.NdArray(
                data=poses.tobytes(),
                dtype=str(poses.dtype),
                shape=list(poses.shape)
            ),
            PCDs=Service_pb2.NdArray(
                data=pcds.tobytes(),
                dtype=str(pcds.dtype),
                shape=list(pcds.shape)
            )
        )

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


class CollaborationServerThread(threading.Thread):                          # 协同感知子系统的Server线程
    def __init__(self, others_info):
        super().__init__()
        self.others_info = others_info

    def run(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
            ('grpc.max_send_message_length', 64 * 1024 * 1024),                     # 设置gRPC 消息的最大发送和接收大小为64MB
            ('grpc.max_receive_message_length', 64 * 1024 * 1024)])
        Service_pb2_grpc.add_CollaborationServiceServicer_to_server(CollaborationService(self.others_info), server)
        server.add_insecure_port('[::]:50052')
        server.start()                              # 非阻塞, 会实例化一个新线程来处理请求
        print("Collaboration Server is up and running on port 50052.")
        try:
            server.wait_for_termination()           # 保持服务器运行直到终止
        except KeyboardInterrupt:
            server.stop(0)                          # 服务器终止
            print("Collaboration Server terminated.")
