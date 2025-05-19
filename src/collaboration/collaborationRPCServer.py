import logging
import threading
from collaboration.collaborationTable import CollaborationTable
from appConfig import AppConfig
import grpc
import time
import numpy as np
from concurrent import futures

from rpc import Service_pb2
from rpc import Service_pb2_grpc


class SharedOthersInfo:
    def __init__(self, ctable: CollaborationTable):
        self.__ids = []               # 初始化为空数组
        self.__timestamps = []        # 初始化为空数组
        self.__poses = np.array([])   # 初始化为空数组
        self.__pcds = np.array([])    # 初始化为空数组
        self.__velocities = np.array([])  # 所有他车的速度
        self.__accelerations = np.array([])  # 所有他车的加速度

        self.__features_lens = []
        self.__voxel_features = np.array([])  # 所有他车体素特征
        self.__voxel_coords = np.array([])  # 所有他车体素坐标
        self.__voxel_num_points = np.array([])  # 所有他车体素点数

        self.__comm_masks = np.array([])

        self.__lock = threading.Lock()

        self.ctable = ctable

    def update_info(self):
        with self.__lock:
            infos = self.ctable.get_all_data()
            self.__ids = [info.id for info in infos]
            self.__timestamps = [info.ts_feat for info in infos]
            self.__poses = [info.lidar_pos for info in infos]
            self.__pcds = [info.pcd for info in infos]
            self.__velocities = [info.speed for info in infos]
            self.__accelerations = [info.acc for info in infos]

            self.__features_lens = np.ndarray([len(infos)])
            self.__voxel_features = np.stack([info.feat['voxel_features'] for info in infos])
            self.__voxel_coords = np.stack([info.feat['voxel_coords'] for info in infos])
            self.__voxel_num_points = np.stack([info.feat['voxel_num_points'] for info in infos])

            self.__comm_masks = np.stack([self.ctable.get_coopmap(info.id).map for info in infos])

    def get_info_copy(self):
        self.update_info()
        with self.__lock:
            ids_copy = self.__ids.copy()
            timestamps_copy = self.__timestamps.copy()
            poses_copy = self.__poses.copy() if self.__poses.size > 0 else self.__poses
            velocities_copy = self.__velocities.copy() if self.__velocities.size > 0 else self.__velocities
            accelerations_copy = self.__accelerations.copy() if self.__accelerations.size > 0 else self.__accelerations

            features_lens_copy = self.__features_lens.copy()

            if self.__voxel_features.size > 0:
                voxel_features_copy = self.__voxel_features.copy()
            else:
                voxel_features_copy = self.__voxel_features

            if self.__voxel_coords.size > 0:
                voxel_coords_copy = self.__voxel_coords.copy()
            else:
                voxel_coords_copy = self.__voxel_coords

            if self.__voxel_features.size > 0:
                voxel_num_points_copy = self.__voxel_num_points.copy()
            else:
                voxel_num_points_copy = self.__voxel_num_points

            info_copy = {'ids': ids_copy,
                         'timestamps': timestamps_copy,
                         'poses': poses_copy,
                         'velocities': velocities_copy,
                         'accelerations': accelerations_copy,
                         'features_lens': features_lens_copy,
                         'voxel_features': voxel_features_copy,
                         'voxel_coords': voxel_coords_copy,
                         'voxel_num_points': voxel_num_points_copy}

            return info_copy

    def get_ids_copy(self):
        self.update_info()
        with self.__lock:
            return self.__ids.copy()

    def get_timestamps_copy(self):
        self.update_info()
        with self.__lock:
            return self.__timestamps.copy()

    def get_poses_copy(self):
        self.update_info()
        with self.__lock:
            poses_copy = self.__poses.copy() if self.__poses.size > 0 else self.__poses
            return self.__ids.copy(), self.__timestamps.copy(), poses_copy

    def get_pcds_copy(self):
        self.update_info()
        with self.__lock:
            pcd_copy = self.__pcds.copy() if self.__pcds.size > 0 else self.__pcds
            return self.__ids.copy(), self.__timestamps.copy(), pcd_copy

    def get_velocities_copy(self):
        self.update_info()
        with self.__lock:
            velocities_copy = self.__velocities.copy() if self.__velocities.size > 0 else self.__velocities
            return self.__ids.copy(), self.__timestamps.copy(), velocities_copy

    def get_accelerations_copy(self):
        self.update_info()
        with self.__lock:
            accelerations_copy = self.__accelerations.copy() if self.__accelerations.size > 0 else self.__accelerations
            return self.__ids.copy(), self.__timestamps.copy(), accelerations_copy

    def get_features_copy(self):
        self.update_info()
        with self.__lock:
            features_lens_copy = self.__features_lens.copy()

            if self.__voxel_features.size > 0:
                voxel_features_copy = self.__voxel_features.copy()
            else:
                voxel_features_copy = self.__voxel_features

            if self.__voxel_coords.size > 0:
                voxel_coords_copy = self.__voxel_coords.copy()
            else:
                voxel_coords_copy = self.__voxel_coords

            if self.__voxel_features.size > 0:
                voxel_num_points_copy = self.__voxel_num_points.copy()
            else:
                voxel_num_points_copy = self.__voxel_num_points

            return self.__ids.copy(), self.__timestamps.copy(), \
                features_lens_copy, voxel_features_copy, voxel_coords_copy, voxel_num_points_copy

    def get_comm_masks_copy(self):
        self.update_info()
        with self.__lock:
            comm_masks_copy = self.__comm_masks.copy() if self.__comm_masks.size > 0 else self.__comm_masks
            return self.__ids.copy(), self.__timestamps.copy(), comm_masks_copy


class CollaborationRPCService(Service_pb2_grpc.CollaborationServiceServicer):  # 协同感知子系统的Service类
    def __init__(self, cfg: AppConfig, others_info: SharedOthersInfo):
        self.cfg = cfg
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
        others_info = self.others_info.get_info_copy()

        return Service_pb2.OthersInfo(  # 序列化并返回所有车辆信息
            ids=others_info['ids'],
            timestamps=others_info['timestamps'],
            poses=Service_pb2.NdArray(
                data=others_info['poses'].tobytes(),
                dtype=str(others_info['poses'].dtype),
                shape=list(others_info['poses'].shape)
            ),
            velocities=Service_pb2.NdArray(
                data=others_info['velocities'].tobytes(),
                dtype=str(others_info['velocities'].dtype),
                shape=list(others_info['velocities'].shape)
            ),
            accelerations=Service_pb2.NdArray(
                data=others_info['accelerations'].tobytes(),
                dtype=str(others_info['accelerations'].dtype),
                shape=list(others_info['accelerations'].shape)
            ),
            features_lens=others_info['features_lens'],
            voxel_features=Service_pb2.NdArray(
                data=others_info['voxel_features'].tobytes(),
                dtype=str(others_info['voxel_features'].dtype),
                shape=list(others_info['voxel_features'].shape)
            ),
            voxel_coords=Service_pb2.NdArray(
                data=others_info['voxel_coords'].tobytes(),
                dtype=str(others_info['voxel_coords'].dtype),
                shape=list(others_info['voxel_coords'].shape)
            ),
            voxel_num_points=Service_pb2.NdArray(
                data=others_info['voxel_num_points'].tobytes(),
                dtype=str(others_info['voxel_num_points'].dtype),
                shape=list(others_info['voxel_num_points'].shape)
            )
        )

    def GetOthersCommMasks(self, request, context):  # 协同感知子系统向其他进程提供“获取所有他车协作图”的服务
        ids = self.others_info.get_ids_copy()
        timestamps = self.others_info.get_timestamps_copy()
        others_comm_masks = self.others_info.get_comm_masks_copy()

        return Service_pb2.OthersCommMasks(  # 序列化并返回所有他车协作图
            ids=ids,
            timestamps=timestamps,
            others_comm_masks=Service_pb2.NdArray(
                data=others_comm_masks.tobytes(),
                dtype=str(others_comm_masks.dtype),
                shape=list(others_comm_masks.shape)
            )
        )


class CollaborationRPCServerThread():                          # 协同感知子系统的Server线程
    def __init__(self, cfg: AppConfig, others_info):
        super().__init__()
        self.cfg = cfg
        self.others_info = others_info
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
            ('grpc.max_send_message_length', 64 * 1024 * 1024),                     # 设置gRPC 消息的最大发送和接收大小为64MB
            ('grpc.max_receive_message_length', 64 * 1024 * 1024)])
        Service_pb2_grpc.add_CollaborationServiceServicer_to_server(CollaborationRPCService(self.cfg, self.others_info), self.server)
        self.stop_event = threading.Event()
        self.run_thread = threading.Thread(target=self.run, name='collaboration rpc server', daemon=True)

    def run(self):
        self.server.add_insecure_port('[::]:50052')
        self.server.start()                              # 非阻塞, 会实例化一个新线程来处理请求
        logging.info("Collaboration Server is up and running on port 50052.")
        try:
            # 等待停止事件或被中断
            while not self.stop_event.is_set():
                self.stop_event.wait(1)  # 每1秒检查一次停止标志
        except KeyboardInterrupt:
            pass
        finally:
            # 优雅地关闭服务器
            if self.server:
                self.server.stop(0.5).wait()

    def start(self):
        self.run_thread.start()

    def close(self):
        self.stop_event.set()  # 设置停止标志