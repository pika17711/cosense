import argparse
import math

import torch
import numpy as np
import open3d as o3d
import cv2

from appConfig import AppConfig
from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils
from opencood.visualization.vis_utils import bbx2oabb, color_encoding
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from utils.sharedInfo import SharedInfo
from utils.perception_utils import get_lidar_pose_and_pcd_from_dataset
from utils.detection_utils import pcd_to_spatial_feature, spatial_feature_to_pred_box, pcd_to_voxel, voxel_to_spatial_feature


def load_model(hypes, model_dir):
    model = train_utils.create_model(hypes)
    # model = point_pillar_where2comm.PointPillarWhere2comm(self.hypes['model']['args'])
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()

    _, model = train_utils.load_saved_model(model_dir, model)
    model.eval()

    return model


def get_shared_info():
    cfg = AppConfig()
    # cfg.model_dir = r'D:\WorkSpace\Python\interopera\opencood\logs\point_pillar_where2comm_2024_10_28_23_24_50'
    # cfg.model_dir = r'D:\WorkSpace\Python\V2V4Real-main\opencood\logs\point_pillar_where2comm_2025_07_20_21_15_26'
    # cfg.model_dir = r'D:\WorkSpace\Python\OpenCOOD-main\opencood\weights\point_pillar_where2comm_2024_10_28_23_24_50'
    cfg.model_dir = r'D:\WorkSpace\Python\interopera\opencood\logs\point_pillar_where2comm_v2v4real'
    opt = argparse.Namespace()
    opt.fusion_method = 'intermediate'
    opt.model_dir = cfg.model_dir

    hypes = yaml_utils.load_yaml(None, opt)

    # ##########################################
    # hypes['model']['args']['where2comm_fusion']['communication']['threshold'] = 0.1
    # ##########################################

    model = load_model(hypes, cfg.model_dir)

    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"模型总参数量: {total_params}")
    # exit()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pre_processor = build_preprocessor(hypes['preprocess'], False)
    post_processor = build_postprocessor(hypes['postprocess'], False)

    shared_info = SharedInfo()
    shared_info.update_hypes(hypes)
    shared_info.update_model(model)
    shared_info.update_device(device)
    shared_info.update_pre_processor(pre_processor)
    shared_info.update_post_processor(post_processor)
    return shared_info


def get_vis():
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True, window_name='Inference_test')
    vis_opt = vis.get_render_option()
    vis_opt.background_color = np.asarray([0, 0, 0])
    vis_opt.point_size = 1.0
    return vis


def show_vis(vis, processed_pcd, pred_box=None, save_vis=False):
    vis.clear_geometries()

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])  # 坐标系
    vis.add_geometry(axis)

    pcd = o3d.geometry.PointCloud()

    # o3d use right-hand coordinate
    # processed_pcd[:, :1] = -processed_pcd[:, :1]

    pcd.points = o3d.utility.Vector3dVector(processed_pcd[:, :3])

    origin_lidar_intcolor = color_encoding(processed_pcd[:, 2], mode='constant')
    pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)

    vis.add_geometry(pcd)
    if pred_box is not None and pred_box.size > 0:
        oabbs_pred = bbx2oabb(pred_box, color=(1, 0, 0), left_hand_coordinate=False)
        for oabb in oabbs_pred:
            vis.add_geometry(oabb)

    # oabbs_gt = get_oabbs_gt(self.shared_info)
    # for oabb in oabbs_gt:
    #     self.vis.add_geometry(oabb)

    # 渲染场景
    # self.vis.poll_events()
    # self.vis.update_renderer()

    # cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
    #
    # # cam.extrinsic = np.array([[1, 0, 0, 0],  # 调整相机位置
    # #                           [0, 1, 0, 0],
    # #                           [0, 0, 1, 0],  # Z值增大=拉远相机
    # #                           [0, 0, 0, 1]])
    # extrinsic = np.array(cam.extrinsic)
    # extrinsic[2, 3] -= 150
    # cam.extrinsic = extrinsic
    # vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

    vis.run()
    if save_vis:
        vis.capture_screen_image('saved.png')
        img = np.asarray(vis.capture_screen_float_buffer(do_render=False))
        img_cv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('img', img_cv)
        cv2.waitKey()



if __name__ == '__main__':
    save_vis = True

    # pcd_file = 'D:\\WorkSpace\\Python\\cosense\\tests\\pcds\\25_07_26\\test11.json'
    pcd_file = 'D:\\Documents\\datasets\\OPV2V\\test_culver_city\\2021_09_03_09_32_17\\311\\006220.pcd'
    # pcd_file = 'D:\\Documents\\datasets\\V2X4Real\\train\\testoutput_CAV_data_2022-03-15-10-09-50_0\\0\\000000.pcd'

    _, pcd = get_lidar_pose_and_pcd_from_dataset(pcd_file)

    pcd_test = pcd.copy()

    # ref_distance = 20.0
    #
    # distances = np.linalg.norm(pcd_test[:, :3], axis=1)
    #
    # # 距离衰减补偿 (平方反比定律)
    # compensated = pcd_test[:, 3] * (distances ** 2) / (ref_distance ** 2)
    #
    # # 截断到物理有效范围 (20-240)
    # clipped = np.clip(pcd_test[:, 3], 20, 240)
    #
    # # 线性映射到0-1
    # normalized = (clipped - 20) / (240 - 20)

    intensity = pcd_test[:, 3]      # pcd_test: [N, 4]  [x, y, z, intensity]
    # print(intensity)
    ######################################################
    # intensity[:] = intensity[:] / 255.0
    # intensity[:] = 1.0

    # intensity[:, 3] = 1 / (1 + np.exp(-intensity))

    shared_info = get_shared_info()
    vis = get_vis()

    # processed_pcd, spatial_feature = pcd_to_spatial_feature(pcd_test, shared_info)
    processed_pcd, voxel = pcd_to_voxel(pcd_test, shared_info)

    # for i in range(len(voxel['voxel_num_points'])):
    #     voxel['voxel_features'][i][:voxel['voxel_num_points'][i], 3] = 0.8

    spatial_feature = voxel_to_spatial_feature(voxel, shared_info)

    pred_box, ego_comm_mask, _, _ = spatial_feature_to_pred_box(spatial_feature, shared_info)
    # print(pred_box.shape)

    show_vis(vis, processed_pcd, pred_box, save_vis)

