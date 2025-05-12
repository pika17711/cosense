import argparse
import time
import numpy as np
from tqdm import tqdm

import torch
import open3d as o3d
from torch.utils.data import DataLoader
import zmq

from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils
from opencood.tools import inference_utils
from opencood.data_utils.datasets import intermediate_fusion_dataset
from opencood.models import point_pillar_where2comm
from opencood.data_utils.datasets import build_dataset
from opencood.utils.transformation_utils import x1_to_x2
from opencood.utils import box_utils
from opencood.utils import eval_utils
from opencood.visualization.vis_utils import color_encoding
import matplotlib.pyplot as plt

import os
import yaml
import re

from detection.detection_server import DetectionServerThread, pcd2feature
from perception.perception_client import PerceptionClient
from collaboration.collaboration_client import CollaborationClient
from utils.sharedInfo import SharedInfo


def fuse_pcd(my_timestamp, my_pcd, my_pose, timestamps, others_poses, others_pcds):
    fused_pcd = [my_pcd]

    for i in range(len(timestamps)):
        # if timestamps[i] == my_timestamp:
        #     transformation_matrix = x1_to_x2(others_poses[i], my_pose)
        #     others_pcds[:, :3] = box_utils.project_points_by_matrix_torch(others_pcds[:, :3], transformation_matrix)
        #     fused_pcd.append(others_pcds)
        transformation_matrix = x1_to_x2(others_poses[i], my_pose)
        others_pcd = others_pcds[i].copy()
        others_pcd[:, :3] = box_utils.project_points_by_matrix_torch(others_pcd[:, :3], transformation_matrix)
        fused_pcd.append(others_pcd)

    fused_pcd = np.vstack(fused_pcd)
    return fused_pcd


class DetectionManager:
    def __init__(self, opt):
        self.opt = opt
        assert self.opt.fusion_method in ['late', 'early', 'intermediate']

        self.hypes = yaml_utils.load_yaml(None, opt)
        self.dataset = self.__load_dataset()
        self.model = self.__load_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        shared_info = SharedInfo()
        shared_info.update_model(self.model)
        shared_info.update_device(self.device)
        shared_info.update_hypes(self.hypes)
        shared_info.update_dataset(self.dataset)
        self.shared_info = shared_info

        self.__grpc_prepare()

        if opt.show_vis:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(visible=True)
            vis_opt = self.vis.get_render_option()
            # vis_opt.background_color = np.asarray([0, 0, 0])
            # vis_opt.point_size = 1.0

    def __load_dataset(self):
        print('Dataset Building')
        opencood_dataset = build_dataset(self.hypes, visualize=True, train=False)
        # opencood_dataset = intermediate_fusion_dataset.IntermediateFusionDataset(self.hypes, visualize=True, train=False)
        # print(f"{len(opencood_dataset)} samples found.")

        return opencood_dataset

    def __load_model(self):
        print('Creating Model')
        model = train_utils.create_model(self.hypes)
        # model = point_pillar_where2comm.PointPillarWhere2comm(self.hypes['model']['args'])
        # we assume gpu is necessary
        if torch.cuda.is_available():
            model.cuda()

        print('Loading Model from checkpoint')
        saved_path = self.opt.model_dir
        _, model = train_utils.load_saved_model(saved_path, model)
        model.eval()

        return model

    def __grpc_prepare(self):
        detection_server_thread = DetectionServerThread(self.shared_info)
        detection_server_thread.setDaemon(True)
        detection_server_thread.start()

        self.perception_client = PerceptionClient()
        self.collaboration_client = CollaborationClient()

    def loop(self):
        record_len = torch.empty(0, dtype=torch.int32)

        pairwise_t_matrix = torch.zeros((1, 5, 5, 4, 4), dtype=torch.float64)

        transformation_matrix = np.eye(4, dtype=np.float32)
        transformation_matrix = torch.from_numpy(transformation_matrix)

        with self.shared_info.dataset_lock:
            anchor_box = self.dataset.post_processor.generate_anchor_box()
        anchor_box = torch.from_numpy(anchor_box)

        ego_data = {'origin_lidar': [],
                    'processed_lidar': {},
                    'record_len': record_len,
                    'pairwise_t_matrix': pairwise_t_matrix,
                    'transformation_matrix': transformation_matrix,
                    'anchor_box': anchor_box}

        batch_data = {'ego': ego_data}

        loop_time = 1
        last_t = 0

        while True:
            t = time.time()
            if t - last_t < loop_time:
                time.sleep(loop_time + last_t - t)
            last_t = t

            my_timestamp, my_pose, my_pcd = self.perception_client.get_my_pose_and_pcd()
            if my_timestamp == -1:
                print("perception子系统未启动")
                continue
            # print(my_pose)
            ids, others_timestamps, others_poses, others_pcds = self.collaboration_client.get_others_poses_and_pcds()
            if ids != -1:
                fused_pcd = fuse_pcd(my_timestamp, my_pcd, my_pose, others_timestamps, others_poses, others_pcds)
            else:
                fused_pcd = my_pcd
            # _, fused_pcd = self.perception_client.get_my_pcd()

            processed_pcd, fused_feature = pcd2feature(fused_pcd, self.hypes)

            processed_pcd = torch.from_numpy(processed_pcd)
            fused_feature['voxel_features'] = torch.from_numpy(fused_feature['voxel_features'])

            voxel_coords = np.pad(fused_feature['voxel_coords'], ((0, 0), (1, 0)), mode='constant', constant_values=0)
            fused_feature['voxel_coords'] = torch.from_numpy(voxel_coords)

            fused_feature['voxel_num_points'] = torch.from_numpy(fused_feature['voxel_num_points'])

            batch_data['ego']['origin_lidar'] = processed_pcd
            batch_data['ego']['processed_lidar'] = fused_feature

            # batch_data['other'] = batch_data['ego']

            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, self.device)
                # if self.opt.fusion_method == 'late':
                #     with self.shared_info.model_lock:
                #         pred_box_tensor, pred_score, gt_box_tensor = \
                #         inference_utils.inference_late_fusion(batch_data, self.model, self.dataset)
                # elif self.opt.fusion_method == 'early':
                #     with self.shared_info.model_lock:
                #         pred_box_tensor, pred_score, gt_box_tensor = \
                #         inference_utils.inference_early_fusion(batch_data, self.model, self.dataset)
                # elif self.opt.fusion_method == 'intermediate':
                #     with self.shared_info.model_lock:
                #         pred_box_tensor, pred_score, gt_box_tensor = \
                #         inference_utils.inference_intermediate_fusion(batch_data, self.model, self.dataset)

                pred_box_tensor, pred_score = \
                    inference_utils.inference_intermediate_fusion(batch_data, self.model, self.dataset)
            print(pred_box_tensor, file=open('result', 'w'))
            if self.opt.show_vis:
                pcd = o3d.geometry.PointCloud()
                origin_lidar = np.asarray(batch_data['ego']['origin_lidar'].cpu())[:, :3]

                pcd.points = o3d.utility.Vector3dVector(origin_lidar)

                # origin_lidar_intcolor = color_encoding(origin_lidar[:, 2], mode='constant')
                # pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)

                pred_box_tensor = pred_box_tensor.cpu()

                # print(pred_box_tensor)

                self.vis.clear_geometries()
                self.vis.add_geometry(pcd)
                for box_points in pred_box_tensor:
                    # 创建一个 LineSet 来连接顶点并显示边界框
                    lines = [
                        [0, 1], [1, 2], [2, 3], [3, 0],  # 上面四个边
                        [4, 5], [5, 6], [6, 7], [7, 4],  # 下面四个边
                        [0, 4], [1, 5], [2, 6], [3, 7]  # 上下四个连接
                    ]
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(box_points)  # 设置顶点
                    line_set.lines = o3d.utility.Vector2iVector(lines)  # 设置边
                    line_set.paint_uniform_color([1, 0, 0])  # 设置边框为红色
                    self.vis.add_geometry(line_set)

                # self.vis.run()

                # 渲染场景
                self.vis.poll_events()
                self.vis.update_renderer()
                save_path = 'vis.png'
                # 截图并保存
                self.vis.capture_screen_image(save_path, do_render=True)
                print(f"Saved visualization to {save_path}")
