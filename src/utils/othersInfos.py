import threading
import time

import numpy as np
from collaboration.collaborationTable import CollaborationTable
from appConfig import AppConfig


class OthersInfos:
    def __init__(self, ctable: CollaborationTable):
        self.__others_infos = {}
        self.__lock = threading.Lock()
        self.__ctable = ctable

    class __CAVInfo:
        def __init__(self):
            self.__info = {'lidar_pose': np.array([]),
                           'ts_lidar_pose': -1,
                           'speed': np.array([]),
                           'ts_spd': -1,
                           'acceleration': np.array([]),
                           'ts_acc': -1,
                           'feature': np.array([]),
                           'ts_feature': -1,
                           'comm_mask': np.array([]),
                           'ts_comm_mask': -1,
                           'pcd': np.array([]),
                           'ts_pcd': -1}

        def update_info(self, lidar_pose=None, ts_lidar_pose=None, speed=None, ts_spd=None,
                        acceleration=None, ts_acc=None, feature=None, ts_feature=None, comm_mask=None, ts_comm_mask=None,
                        pcd=None, ts_pcd=None):
            cav_info = {
                'lidar_pose': lidar_pose,
                'ts_lidar_pose': ts_lidar_pose,
                'speed': speed,
                'ts_spd': ts_spd,
                'acceleration': acceleration,
                'ts_acc': ts_acc,
                'feature': feature,
                'ts_feature': ts_feature,
                'comm_mask': comm_mask,
                'ts_comm_mask': ts_comm_mask,
                'pcd': pcd,
                'ts_pcd': ts_pcd
            }

            self.update_info_dict(cav_info)

        def update_info_dict(self, cav_info):
            if cav_info is not None:
                for key, value in cav_info.items():
                    if key in self.__info and value is not None:
                        self.__info[key] = value

        def get_info(self):
            info_copy = self.__info.copy()
            return info_copy

    def __update_info(self, cav_id=None, lidar_pose=None, ts_lidar_pose=None, speed=None, ts_spd=None,
                      acceleration=None, ts_acc=None, feature=None, ts_feature=None, comm_mask=None, ts_comm_mask=None,
                      pcd=None, ts_pcd=None):
        if cav_id is None:
            return
        with self.__lock:
            if cav_id not in self.__others_infos:
                self.__others_infos[cav_id] = self.__CAVInfo()

            self.__others_infos[cav_id].update_info(lidar_pose=lidar_pose, ts_lidar_pose=ts_lidar_pose,
                                                    speed=speed, ts_spd=ts_spd,
                                                    acceleration=acceleration, ts_acc=ts_acc,
                                                    feature=feature, ts_feature=ts_feature,
                                                    comm_mask=comm_mask, ts_comm_mask=ts_comm_mask,
                                                    pcd=pcd, ts_pcd=ts_pcd)

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
        # infos = self.__ctable.pop_all_data()

        others_infos = {}
        for info in infos:
            cav_id = info.id

            feature = info.feat['spatial_feature']
            comm_mask = info.feat.get('comm_mask')

            cav_info = {'lidar_pose': info.lidar_pos,
                        'ts_lidar_pose': info.ts_lidar_pos,
                        'speed': info.speed,
                        'ts_spd': info.ts_spd,
                        'acceleration': info.acc,
                        'ts_acc': info.ts_acc,
                        'feature': feature,
                        'ts_feature': info.ts_feat,
                        'comm_mask': comm_mask,
                        'pcd': info.pcd,
                        'ts_pcd': info.ts_pcd}
            others_infos[cav_id] = cav_info

        self.update_infos(others_infos)

    def get_infos(self):
        self.update_infos_ctable()
        infos_copy = {}
        with self.__lock:
            for cav_id, cav_info in self.__others_infos.items():
                infos_copy[cav_id] = cav_info.get_info()
        return infos_copy

    def pop_infos(self):
        self.update_infos_ctable()
        infos_copy = {}
        with self.__lock:
            for cav_id, cav_info in self.__others_infos.items():
                infos_copy[cav_id] = cav_info.get_info()
            self.__others_infos = {}
        return infos_copy
