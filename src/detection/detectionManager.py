import logging
import time
from appConfig import AppConfig
import numpy as np

import torch
import open3d as o3d

from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.visualization.vis_utils import bbx2oabb, color_encoding
# from tests.detection.utils_test import get_oabbs_gt
from opencood.utils import box_utils
from opencood.utils.transformation_utils import gps_to_utm_transformation, gps_to_enu_transformation

from utils.detection_utils import fuse_spatial_feature, pcd_to_spatial_feature, spatial_feature_to_pred_box, \
                                  get_features_from_cav_infos
from detection.detectionRPCServer import DetectionServerThread
from perception.perceptionRPCClient import PerceptionRPCClient
from collaboration.collaborationRPCClient import CollaborationRPCClient
from utils.sharedInfo import SharedInfo


class DetectionManager:
    def __init__(self, opt, cfg: AppConfig):
        self.opt = opt
        assert self.opt.fusion_method in ['late', 'early', 'intermediate']
        self.cfg = cfg
        self.hypes = yaml_utils.load_yaml(None, opt)
        self.model = self.__load_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pre_processor = build_preprocessor(self.hypes['preprocess'], False)
        self.post_processor = build_postprocessor(self.hypes['postprocess'], False)

        shared_info = SharedInfo()
        shared_info.update_hypes(self.hypes)
        shared_info.update_model(self.model)
        shared_info.update_device(self.device)
        shared_info.update_pre_processor(self.pre_processor)
        shared_info.update_post_processor(self.post_processor)
        self.shared_info = shared_info

        self.running = False
        self.__grpc_prepare()

        if opt.show_vis:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(visible=True, window_name='Detection')
            vis_opt = self.vis.get_render_option()
            vis_opt.background_color = np.asarray([0, 0, 0])
            vis_opt.point_size = 1.0

    def __load_model(self):
        logging.info('Creating Model')
        model = train_utils.create_model(self.hypes)
        # model = point_pillar_where2comm.PointPillarWhere2comm(self.hypes['model']['args'])
        # we assume gpu is necessary
        if torch.cuda.is_available():
            model.cuda()

        logging.info('Loading Model from checkpoint')
        saved_path = self.opt.model_dir
        _, model = train_utils.load_saved_model(saved_path, model)
        model.eval()

        return model

    def __grpc_prepare(self):
        self.detection_rpc_server = DetectionServerThread(self.cfg, self.shared_info)

        self.perception_client = PerceptionRPCClient(self.cfg)
        self.collaboration_client = CollaborationRPCClient(self.cfg)

    def start(self):
        self.running = True
        self.detection_rpc_server.start()
        self.__loop()

    def __loop(self):
        loop_time = 0.3
        last_t = time.time() - loop_time

        while self.running:
            t = time.time()
            if t - last_t < loop_time:
                time.sleep(loop_time + last_t - t)
            t = time.time()
            print(f'last loop time: {t - last_t}s')
            last_t = t

            logging.info("融合检测进行中...")
            # 获取自车pcd并处理得到自车特征
            my_lidar_pose, ts_lidar_pose, my_pcd, ts_pcd = self.perception_client.get_my_lidar_pose_and_pcd()
            if my_lidar_pose is None:
                # RPC 错误 已经在RPC调用中输出过日志
                continue
            self.shared_info.update_perception_info(lidar_pose=my_lidar_pose, pcd=my_pcd)
            processed_pcd, my_spatial_feature = pcd_to_spatial_feature(my_pcd, self.shared_info)
            print('my_spatial_feature.shape: ' + str(my_spatial_feature.shape))
            # 获取他车特征
            cav_infos = self.collaboration_client.get_others_infos()
            if self.cfg.collaboration_pcd_debug:
                others_lidar_poses_and_pcds = self.collaboration_client.get_others_lidar_poses_and_pcds()

            spatial_features, comm_masked_features = get_features_from_cav_infos(cav_infos)

            fused_spatial_feature = fuse_spatial_feature(my_spatial_feature, spatial_features)
            print('fused_spatial_feature.shape: ' + str(fused_spatial_feature.shape))

            # 根据融合后的特征得到检测框
            pred_box, ego_comm_mask = spatial_feature_to_pred_box(fused_spatial_feature, self.shared_info, comm_masked_features)
            self.shared_info.update_pred_box(pred_box)
            self.shared_info.update_comm_mask(ego_comm_mask)

            projected_others_pcds = None
            if self.cfg.collaboration_pcd_debug:
                projected_others_pcds = []
                if others_lidar_poses_and_pcds is not None:
                    for cav_id, cav_lidar_pose_and_pcd in others_lidar_poses_and_pcds.items():
                        cav_lidar_pose = cav_lidar_pose_and_pcd['lidar_pose']
                        cav_pcd = cav_lidar_pose_and_pcd['pcd']

                        transformation_matrix = gps_to_utm_transformation(cav_lidar_pose, my_lidar_pose)
                        cav_pcd[:, :3] = box_utils.project_points_by_matrix_torch(cav_pcd[:, :3], transformation_matrix)
                        projected_others_pcds.append(cav_pcd)

            if self.opt.show_vis:
                self.__show_vis(processed_pcd, pred_box, projected_others_pcds)

    def __show_vis(self, processed_pcd, pred_box=None, projected_others_pcds=None):
        self.vis.clear_geometries()

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])  # 坐标系
        self.vis.add_geometry(axis)

        pcd = o3d.geometry.PointCloud()

        left_hand_coordinate = self.cfg.perception_debug and self.cfg.perception_debug_data_from_OPV2V
        if left_hand_coordinate:
            processed_pcd[:, :1] = -processed_pcd[:, :1]

        pcd.points = o3d.utility.Vector3dVector(processed_pcd[:, :3])

        origin_lidar_intcolor = color_encoding(processed_pcd[:, 2], mode='constant')
        pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)

        self.vis.add_geometry(pcd)

        if projected_others_pcds is not None:
            for projected_pcd in projected_others_pcds:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(projected_pcd[:, :3])
                red_color = np.array([1.0, 0, 0])  # RGB值范围[0,1]
                pcd.colors = o3d.utility.Vector3dVector(np.tile(red_color, (len(pcd.points), 1)))  # 复制颜色到所有点

                self.vis.add_geometry(pcd)

        if pred_box is not None and pred_box.size > 0:
            oabbs_pred = bbx2oabb(pred_box, color=(1, 0, 0), left_hand_coordinate=left_hand_coordinate)
            for oabb in oabbs_pred:
                self.vis.add_geometry(oabb)

        # oabbs_gt = get_oabbs_gt(self.shared_info)
        # for oabb in oabbs_gt:
        #     self.vis.add_geometry(oabb)

        # 渲染场景
        self.vis.poll_events()
        self.vis.update_renderer()

        # self.vis.run()

        # save_path = 'vis.png'
        # # 截图并保存
        # self.vis.capture_screen_image(save_path)
        # print(f"Saved visualization to {save_path}")

    def close(self):
        self.running = False
        self.detection_rpc_server.close()
