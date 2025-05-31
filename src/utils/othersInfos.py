import threading
import time

import numpy as np
from collaboration.collaborationTable import CollaborationTable
from appConfig import AppConfig


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

            self.__features_lens = [info.feat['voxel_features'].shape[0] for info in infos]
            self.__voxel_features = np.stack([info.feat['voxel_features'] for info in infos])
            self.__voxel_coords = np.stack([info.feat['voxel_coords'] for info in infos])
            self.__voxel_num_points = np.stack([info.feat['voxel_num_points'] for info in infos])

            self.__comm_masks = np.stack([self.ctable.get_coopmap(info.id).map for info in infos])


class OthersInfos:
    def __init__(self, ctable: CollaborationTable):
        self.__others_infos = {}
        self.__lock = threading.Lock()
        self.__ctable = ctable

    class __CAVInfo:
        def __init__(self):
            self.__info = {'lidar_pose': np.array([]),
                           'ts_lidar_pose': -1,
                           'velocity': np.array([]),
                           'ts_v': -1,
                           'acceleration': np.array([]),
                           'ts_a': -1,
                           'feature': np.array([]),
                           'ts_feature': -1,
                           'comm_mask': np.array([]),
                           'ts_comm_mask': -1}

        def update_info(self, lidar_pose=None, ts_lidar_pose=None, velocity=None, ts_v=None,
                        acceleration=None, ts_a=None, feature=None, ts_feature=None, comm_mask=None, ts_comm_mask=None):
            params = {
                'lidar_pose': lidar_pose,
                'ts_lidar_pose': ts_lidar_pose,
                'velocity': velocity,
                'ts_v': ts_v,
                'acceleration': acceleration,
                'ts_a': ts_a,
                'feature': feature,
                'ts_feature': ts_feature,
                'comm_mask': comm_mask,
                'ts_comm_mask': ts_comm_mask,
            }

            for key, value in params.items():
                if value is not None:
                    self.__info[key] = value

        def update_info_dict(self, cav_info):
            if cav_info is not None:
                for key, value in cav_info.items():
                    if key in self.__info and value is not None:
                        self.__info[key] = value

        def get_info(self):
            info_copy = self.__info.copy()
            return info_copy

    def __update_info(self, cav_id=None, lidar_pose=None, ts_lidar_pose=None, velocity=None, ts_v=None,
                      acceleration=None, ts_a=None, feature=None, ts_feature=None, comm_mask=None, ts_comm_mask=None):
        if cav_id is None:
            return
        with self.__lock:
            if cav_id not in self.__others_infos:
                self.__others_infos[cav_id] = self.__CAVInfo()

            self.__others_infos[cav_id].update_info(lidar_pose=lidar_pose, ts_lidar_pose=ts_lidar_pose,
                                                    velocity=velocity, ts_v=ts_v,
                                                    acceleration=acceleration, ts_a=ts_a,
                                                    feature=feature, ts_feature=ts_feature,
                                                    comm_mask=comm_mask, ts_comm_mask=ts_comm_mask)

    def __update_info_dict(self, cav_id=None, cav_info=None):
        if cav_id is None or cav_info is None:
            return
        if cav_id not in self.__others_infos:
            self.__others_infos[cav_id] = self.__CAVInfo()

        self.__others_infos[cav_id].update_info_dict(cav_info)

    def update_infos(self, others_infos):
        if others_infos is None:
            return
        with self.__lock:
            self.__others_infos = \
                {cav_id: cav_info for cav_id, cav_info in self.__others_infos.items() if cav_id in others_infos}
            for cav_id, cav_info in others_infos.items():
                self.__update_info_dict(cav_id, cav_info)

    def update_infos_ctable(self):
        infos = self.__ctable.get_all_data()
        others_infos = {}
        for info in infos:
            cav_id = info.id
            cav_info = {'lidar_pose': info.lidar_pos,
                        'ts_lidar_pose': info.ts_lidar_pos,
                        'velocity': info.speed,
                        'ts_v': info.ts_speed,
                        'acceleration': info.acc,
                        'ts_a': info.ts_acc,
                        'feature': info.feat,
                        'ts_feature': info.ts_feat,
                        'comm_mask': self.__ctable.get_coopmap(info.id).map,
                        'ts_comm_mask': int(time.time())}
            others_infos[cav_id] = cav_info

        self.update_infos(others_infos)

    def get_infos(self):
        infos_copy = {}
        with self.__lock:
            for cav_id, cav_info in self.__others_infos.items():
                infos_copy[cav_id] = cav_info.get_info()
        return infos_copy
