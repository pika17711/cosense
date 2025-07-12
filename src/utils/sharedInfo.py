import threading
import numpy as np


class SharedInfo:
    def __init__(self):
        self.__pcd = np.array([])  # 初始化为空数组
        self.__ts_pcd = 0
        self.__lidar_pose = np.array([])  # 初始化为空数组
        self.__ts_lidar_pose = 0
        self.__speed = 0.0
        self.__ts_speed = 0
        self.__acceleration = 0.0
        self.__ts_acc = 0
        self.__extrinsic_matrix = np.array([])  # 初始化为空数组
        self.__perception_lock = threading.Lock()
        self.__extrinsic_matrix_lock = threading.Lock()

        self.__hypes = None
        self.__model = None
        self.__device = None
        self.__pre_processor = None
        self.__post_processor = None
        self.__fused_feature = {}  # 初始化为空字典
        self.__fused_comm_mask = np.array([])  # 初始化为空数组
        self.__pred_box = np.array([])  # 初始化为空数组
        self.__ts_pred_box = 0
        self.__feature = np.array([])  # 初始化为空数组
        self.__ts_feature = 0
        self.__conf_map = np.array([])  # 初始化为空数组
        self.__ts_conf_map = 0
        self.__comm_mask = np.array([])  # 初始化为空数组
        self.__ts_comm_mask = 0
        self.model_lock = threading.Lock()
        self.pre_processor_lock = threading.Lock()
        self.post_processor_lock = threading.Lock()
        self.__fused_feature_lock = threading.Lock()
        self.__fused_comm_mask_lock = threading.Lock()
        self.__pred_box_lock = threading.Lock()
        self.__feature_lock = threading.Lock()
        self.__conf_map_lock = threading.Lock()
        self.__comm_mask_lock = threading.Lock()

    def update_perception_info(self,
                               pcd=None,
                               lidar_pose=None,
                               speed=None,
                               acceleration=None):
        with self.__perception_lock:
            self.__pcd = pcd
            self.__lidar_pose = lidar_pose
            self.__speed = speed
            self.__acceleration = acceleration

    def update_perception_info_dict(self, perception_info):
        pcd = perception_info.get('pcd', None)
        lidar_pose = perception_info.get('lidar_pose', None)
        speed = perception_info.get('speed', None)
        acceleration = perception_info.get('acceleration', None)

        self.update_perception_info(pcd=pcd, lidar_pose=lidar_pose, speed=speed, acceleration=acceleration)

    def update_extrinsic_matrix(self, extrinsic_matrix):
        with self.__extrinsic_matrix_lock:
            self.__extrinsic_matrix = extrinsic_matrix  # 线程安全更新

    def update_hypes(self, hypes):
        self.__hypes = hypes

    def update_model(self, model):
        self.__model = model

    def update_device(self, device):
        self.__device = device

    def update_pre_processor(self, pre_processor):
        self.__pre_processor = pre_processor

    def update_post_processor(self, post_processor):
        self.__post_processor = post_processor

    def update_fused_feature(self, fused_feature):
        with self.__fused_feature_lock:
            self.__fused_feature = fused_feature  # 线程安全更新

    def update_fused_comm_mask(self, fused_comm_mask):
        with self.__fused_comm_mask_lock:
            self.__fused_comm_mask = fused_comm_mask  # 线程安全更新

    def update_pred_box(self, pred_box):
        with self.__pred_box_lock:
            self.__pred_box = pred_box  # 线程安全更新

    def update_feature(self, feature):
        with self.__feature_lock:
            self.__feature = feature  # 线程安全更新

    def update_conf_map(self, conf_map):
        with self.__conf_map_lock:
            self.__conf_map = conf_map  # 线程安全更新

    def update_comm_mask(self, comm_mask):
        with self.__comm_mask_lock:
            self.__comm_mask = comm_mask  # 线程安全更新

    def get_pcd_copy(self):
        with self.__perception_lock:
            return self.__pcd.copy() if self.__pcd is not None else self.__pcd

    def get_lidar_pose_copy(self):
        with self.__perception_lock:
            return self.__lidar_pose.copy()

    def get_speed_copy(self):
        with self.__perception_lock:
            return self.__speed

    def get_acceleration_copy(self):
        with self.__perception_lock:
            return self.__acceleration

    def get_extrinsic_matrix_copy(self):
        with self.__extrinsic_matrix_lock:
            return self.__extrinsic_matrix.copy()

    def get_hypes(self):
        return self.__hypes

    def get_model(self):
        return self.__model

    def get_device(self):
        return self.__device

    def get_pre_processor(self):
        return self.__pre_processor

    def get_post_processor(self):
        return self.__post_processor

    def get_fused_feature_copy(self):
        with self.__fused_feature_lock:
            return self.__fused_feature.copy()

    def get_fused_comm_mask_copy(self):
        with self.__fused_comm_mask_lock:
            return self.__fused_comm_mask.copy()

    def get_pred_box_copy(self):
        with self.__pred_box_lock:
            return self.__pred_box.copy()

    def get_feature_copy(self):
        with self.__feature_lock:
            return self.__feature.copy()

    def get_conf_map_copy(self):
        with self.__conf_map_lock:
            return self.__conf_map.copy()

    def get_comm_mask_copy(self):
        with self.__comm_mask_lock:
            return self.__comm_mask.copy()
