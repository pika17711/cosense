import threading
import numpy as np


class SharedInfo:
    def __init__(self):
        self.__pre_processor = None
        self.__model = None
        self.__device = None
        self.__hypes = None
        self.__dataset = None
        self.__fused_feature = {}  # 初始化为空字典
        self.__fused_comm_mask = np.array([])  # 初始化为空数组
        self.__pred_box = np.array([])  # 初始化为空数组
        self.__pcd = np.array([])  # 初始化为空数组
        self.__feature = {}  # 初始化为空字典
        self.__conf_map = np.array([])  # 初始化为空数组
        self.__comm_mask = np.array([])  # 初始化为空数组
        self.__pose = np.array([])  # 初始化为空数组
        self.__velocity = np.array([])  # 初始化为空数组
        self.__acceleration = np.array([])  # 初始化为空数组
        self.__extrinsic_matrix = np.array([])  # 初始化为空数组

        self.pre_processor_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.dataset_lock = threading.Lock()
        self.__fused_feature_lock = threading.Lock()
        self.__fused_comm_mask_lock = threading.Lock()
        self.__pred_box_lock = threading.Lock()
        self.__pcd_lock = threading.Lock()
        self.__feature_lock = threading.Lock()
        self.__conf_map_lock = threading.Lock()
        self.__comm_mask_lock = threading.Lock()
        self.__pose_lock = threading.Lock()
        self.__velocity_lock = threading.Lock()
        self.__acceleration_lock = threading.Lock()
        self.__extrinsic_matrix_lock = threading.Lock()

    def update_pre_processor(self, pre_processor):
        self.__pre_processor = pre_processor

    def update_model(self, model):
        self.__model = model

    def update_device(self, device):
        self.__device = device

    def update_hypes(self, hypes):
        self.__hypes = hypes

    def update_dataset(self, dataset):
        self.__dataset = dataset

    def update_fused_feature(self, fused_feature):
        with self.__fused_feature_lock:
            self.__fused_feature = fused_feature  # 线程安全更新

    def update_fused_comm_mask(self, fused_comm_mask):
        with self.__fused_comm_mask_lock:
            self.__fused_comm_mask = fused_comm_mask  # 线程安全更新

    def update_pred_box(self, pred_box):
        with self.__pred_box_lock:
            self.__pred_box = pred_box  # 线程安全更新

    def update_pcd(self, pcd):
        with self.__pcd_lock:
            self.__pcd = pcd  # 线程安全更新

    def update_feature(self, feature):
        with self.__feature_lock:
            self.__feature = feature  # 线程安全更新

    def update_conf_map(self, conf_map):
        with self.__conf_map_lock:
            self.__conf_map = conf_map  # 线程安全更新

    def update_comm_mask(self, comm_mask):
        with self.__comm_mask_lock:
            self.__comm_mask = comm_mask  # 线程安全更新

    def update_pose(self, pose):
        with self.__pose_lock:
            self.__pose = pose  # 线程安全更新

    def update_velocity(self, velocity):
        with self.__velocity_lock:
            self.__velocity = velocity  # 线程安全更新

    def update_acceleration(self, acceleration):
        with self.__acceleration_lock:
            self.__acceleration = acceleration  # 线程安全更新

    def update_extrinsic_matrix(self, extrinsic_matrix):
        with self.__extrinsic_matrix_lock:
            self.__extrinsic_matrix = extrinsic_matrix  # 线程安全更新

    def get_pre_processor(self):
        return self.__pre_processor

    def get_model(self):
        return self.__model

    def get_device(self):
        return self.__device

    def get_hypes(self):
        return self.__hypes

    def get_dataset(self):
        return self.__dataset

    def get_fused_feature_copy(self):
        with self.__fused_feature_lock:
            return self.__fused_feature

    def get_fused_comm_mask_copy(self):
        with self.__fused_comm_mask_lock:
            return self.__fused_comm_mask.copy() if self.__fused_comm_mask.size > 0 else self.__fused_comm_mask

    def get_pred_box_copy(self):
        with self.__pred_box_lock:
            return self.__pred_box.copy() if self.__pred_box.size > 0 else self.__pred_box

    def get_pcd_copy(self):
        with self.__pcd_lock:
            return self.__pcd.copy() if self.__pcd.size > 0 else self.__pcd

    def get_feature_copy(self):
        with self.__feature_lock:
            return self.__feature

    def get_conf_map_copy(self):
        with self.__conf_map_lock:
            return self.__conf_map.copy() if self.__conf_map.size > 0 else self.__conf_map

    def get_comm_mask_copy(self):
        with self.__comm_mask_lock:
            return self.__comm_mask.copy() if self.__comm_mask.size > 0 else self.__comm_mask

    def get_pose_copy(self):
        with self.__pose_lock:
            return self.__pose.copy() if self.__pose.size > 0 else self.__pose

    def get_velocity_copy(self):
        with self.__velocity_lock:
            return self.__velocity.copy() if self.__velocity.size > 0 else self.__velocity

    def get_acceleration_copy(self):
        with self.__acceleration_lock:
            return self.__acceleration.copy() if self.__acceleration.size > 0 else self.__acceleration

    def get_extrinsic_matrix_copy(self):
        with self.__extrinsic_matrix_lock:
            return self.__extrinsic_matrix.copy() if self.__extrinsic_matrix.size > 0 else self.__extrinsic_matrix
