import threading
import numpy as np


class SharedInfo:
    def __init__(self):
        self.__pcd = np.array([])  # 初始化为空数组
        self.__ts_pcd = 0
        self.__lidar_pose = np.array([])  # 初始化为空数组
        self.__ts_lidar_pose = 0
        self.__speed = np.array([])  # 初始化为空数组
        self.__ts_speed = 0
        self.__acceleration = np.array([])  # 初始化为空数组    
        self.__ts_acc = 0
        self.__extrinsic_matrix = np.array([])  # 初始化为空数组
        self.__perception_info_lock = threading.Lock()
        self.__extrinsic_matrix_lock = threading.Lock()

        self.__hypes = None
        self.__model = None
        self.__device = None
        self.__pre_processor = None
        self.__post_processor = None
        self.__fused_feature = np.array([])  # 初始化为空数组
        self.__fused_comm_mask = np.array([])  # 初始化为空数组
        self.__pred_box = np.array([])  # 初始化为空数组
        self.__ts_pred_box = 0
        self.__ego_feature = np.array([])  # 初始化为空数组
        self.__ts_ego_feature = 0
        self.__conf_map = np.array([])  # 初始化为空数组
        self.__ts_conf_map = 0
        self.__ego_comm_mask = np.array([])  # 初始化为空数组
        self.__ts_ego_comm_mask = 0
        self.model_lock = threading.Lock()
        self.pre_processor_lock = threading.Lock()
        self.post_processor_lock = threading.Lock()
        self.__fused_comm_mask_lock = threading.Lock()
        self.__pred_box_lock = threading.Lock()
        self.__conf_map_lock = threading.Lock()
        self.__ego_comm_mask_lock = threading.Lock()
        self.__pcd_img = np.array([])
        self.__others_comm_mask = np.array([])
        self.__presentation_info_lock = threading.Lock()

    def update_perception_info(self,
                               pcd=None,
                               lidar_pose=None,
                               speed=None,
                               acceleration=None):
        with self.__perception_info_lock:
            if pcd is not None:
                self.__pcd = pcd

            if lidar_pose is not None:
                self.__lidar_pose = lidar_pose

            if speed is not None:
                self.__speed = speed

            if acceleration is not None:
                self.__acceleration = acceleration

    def update_perception_info_dict(self, perception_info):
        pcd = perception_info.get('pcd', None)
        lidar_pose = perception_info.get('lidar_pose', None)
        speed = perception_info.get('speed', None)
        acceleration = perception_info.get('acceleration', None)

        self.update_perception_info(pcd=pcd, lidar_pose=lidar_pose, speed=speed, acceleration=acceleration)

    def update_presentation_info(self, 
                                 lidar_pose=None, speed=None,
                                 ego_comm_mask=None,
                                 pcd_img=None, others_comm_mask=None, ego_feature=None, fused_feature=None):
        with self.__perception_info_lock:
            if lidar_pose is not None:
                self.__lidar_pose = lidar_pose
            if speed is not None:
                self.__speed = speed

        with self.__ego_comm_mask_lock:
            if ego_comm_mask is not None:
                self.__ego_comm_mask = ego_comm_mask
        
        with self.__presentation_info_lock:
            if pcd_img is not None:
                self.__pcd_img = pcd_img
            if others_comm_mask is not None:
                self.__others_comm_mask = others_comm_mask
            if ego_feature is not None:
                self.__ego_feature = ego_feature
            if fused_feature is not None:
                self.__fused_feature = fused_feature

    def update_presentation_info_dict(self, presentation_info: dict):
        lidar_pose = presentation_info.get('lidar_pose', None)
        speed = presentation_info.get('speed', None)
        pcd_img = presentation_info.get('pcd_img', None)
        ego_comm_mask = presentation_info.get('ego_comm_mask', None)
        others_comm_mask = presentation_info.get('others_comm_mask', None)
        ego_feature = presentation_info.get('ego_feature', None)
        fused_feature = presentation_info.get('fused_feature', None)

        self.update_presentation_info(lidar_pose=lidar_pose, speed=speed,
                                      ego_comm_mask=ego_comm_mask,
                                      pcd_img=pcd_img, others_comm_mask=others_comm_mask,
                                      ego_feature=ego_feature, fused_feature=fused_feature)

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

    # def update_fused_feature(self, fused_feature):
    #     with self.__fused_feature_lock:
    #         self.__fused_feature = fused_feature  # 线程安全更新

    def update_fused_comm_mask(self, fused_comm_mask):
        with self.__fused_comm_mask_lock:
            self.__fused_comm_mask = fused_comm_mask  # 线程安全更新

    def update_pred_box(self, pred_box):
        with self.__pred_box_lock:
            self.__pred_box = pred_box  # 线程安全更新

    def update_conf_map(self, conf_map):
        with self.__conf_map_lock:
            self.__conf_map = conf_map  # 线程安全更新

    def update_ego_comm_mask(self, ego_comm_mask):
        with self.__ego_comm_mask_lock:
            self.__ego_comm_mask = ego_comm_mask  # 线程安全更新

    def get_pcd_copy(self):
        with self.__perception_info_lock:
            return self.__pcd.copy() if self.__pcd is not None else self.__pcd

    def get_lidar_pose_copy(self):
        with self.__perception_info_lock:
            return self.__lidar_pose.copy() if self.__lidar_pose is not None else self.__lidar_pose

    def get_speed_copy(self):
        with self.__perception_info_lock:
            return self.__speed.copy() if self.__speed is not None else self.__speed

    def get_acceleration_copy(self):
        with self.__perception_info_lock:
            return self.__acceleration.copy() if self.__acceleration is not None else self.__acceleration

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

    # def get_fused_feature_copy(self):
    #     with self.__fused_feature_lock:
    #         return self.__fused_feature.copy()

    def get_fused_comm_mask_copy(self):
        with self.__fused_comm_mask_lock:
            return self.__fused_comm_mask.copy()

    def get_pred_box_copy(self):
        with self.__pred_box_lock:
            return self.__pred_box.copy()

    # def get_ego_feature_copy(self):
    #     with self.__feature_lock:
    #         return self.__ego_feature.copy()

    def get_conf_map_copy(self):
        with self.__conf_map_lock:
            return self.__conf_map.copy()

    def get_ego_comm_mask_copy(self):
        with self.__ego_comm_mask_lock:
            return self.__ego_comm_mask.copy()

    def get_presentation_info_copy(self) -> dict:
        presentation_info = {}
        with self.__perception_info_lock:
            presentation_info['lidar_pose'] = self.__lidar_pose.copy()
            presentation_info['speed'] = self.__speed.copy()

        with self.__presentation_info_lock:
            presentation_info['pcd_img'] = self.__pcd_img.copy()
            presentation_info['others_comm_mask'] = self.__others_comm_mask.copy()
            presentation_info['ego_feature'] = self.__ego_feature.copy()
            presentation_info['fused_feature'] = self.__fused_feature.copy()

        with self.__ego_comm_mask_lock:
            presentation_info['ego_comm_mask'] = self.__ego_comm_mask.copy()

        return presentation_info

