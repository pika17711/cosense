import argparse
import time
import numpy as np
from tqdm import tqdm

import torch
import open3d as o3d
from torch.utils.data import DataLoader
import zmq

import yaml_utils
from opencood.tools import train_utils
import inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils
import matplotlib.pyplot as plt

import os
import yaml
import re

import intermediate_fusion_dataset
import point_pillar_where2comm

from detection.detection_server import DetectionServerThread, SharedInfo, pcd2feature
from perception.perception_client import PerceptionClient

class DetectionManager:
    def __init__(self):
        self.hypes = self.__load_hypes()
        dataset = self.__load_dataset()
        model = self.__load_model()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.shared_info = SharedInfo(model, device, self.hypes, dataset)

        self.__grpc_prepare()

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(visible=True)

    def __load_hypes(self):
        self.opt = argparse.Namespace()
        self.opt.fusion_method = 'intermediate'
        self.opt.model_dir = '../opencood/logs/point_pillar_where2comm_2024_10_28_23_24_50'

        assert self.opt.fusion_method in ['late', 'early', 'intermediate']

        hypes = yaml_utils.load_yaml(None, self.opt)

        return hypes

    def __load_dataset(self):
        print('Dataset Building')
        opencood_dataset = intermediate_fusion_dataset.IntermediateFusionDataset(self.hypes, visualize=True, train=False)
        print(f"{len(opencood_dataset)} samples found.")

        return opencood_dataset

    def __load_model(self):
        print('Creating Model')
        # model = train_utils.create_model(hypes)
        model = point_pillar_where2comm.PointPillarWhere2comm(self.hypes['model']['args'])
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

    def loop(self):
        record_len = torch.empty(0, dtype=torch.int32)

        pairwise_t_matrix = torch.zeros((1, 5, 5, 4, 4), dtype=torch.float64)

        transformation_matrix = np.eye(4, dtype=np.float32)
        transformation_matrix = torch.from_numpy(transformation_matrix)

        with self.shared_info.dataset_lock:
            anchor_box = self.shared_info.dataset.post_processor.generate_anchor_box()
        anchor_box = torch.from_numpy(anchor_box)

        ego_data = {'processed_lidar': {},
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

            timestamp, my_pcd = self.perception_client.get_my_pcd()
            processed_pcd, my_feature = pcd2feature(my_pcd, self.shared_info.hypes)

            processed_pcd = torch.from_numpy(processed_pcd)
            my_feature['voxel_features'] = torch.from_numpy(my_feature['voxel_features'])

            voxel_coords = np.pad(my_feature['voxel_coords'], ((0, 0), (1, 0)), mode='constant', constant_values=0)
            my_feature['voxel_coords'] = torch.from_numpy(voxel_coords)

            my_feature['voxel_num_points'] = torch.from_numpy(my_feature['voxel_num_points'])

            batch_data['ego']['origin_lidar'] = processed_pcd
            batch_data['ego']['processed_lidar'] = my_feature

            batch_data['other'] = batch_data['ego']

            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, self.shared_info.device)

                with self.shared_info.model_lock:
                    # pred_box_tensor, pred_score, gt_box_tensor, comm_masks = \
                    #     inference_utils.inference_intermediate_fusion(batch_data, model, opencood_dataset)
                    pred_box_tensor, pred_score, gt_box_tensor = \
                        inference_utils.inference_late_fusion(batch_data, self.shared_info.model, self.shared_info.dataset)

            pcd = o3d.geometry.PointCloud()
            origin_lidar = np.asarray(batch_data['ego']['origin_lidar'].cpu())[:, :3]

            pcd.points = o3d.utility.Vector3dVector(origin_lidar)
            pred_box_tensor = pred_box_tensor.cpu()

            print(pred_box_tensor)

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

            # 渲染场景
            self.vis.poll_events()
            self.vis.update_renderer()
            save_path = 'vis.png'
            # 截图并保存
            self.vis.capture_screen_image(save_path)
            print(f"Saved visualization to {save_path}")