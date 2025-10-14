import logging
import os
import time
import threading
import open3d as o3d
from appConfig import AppConfig
from collections import deque

from perception.perceptionRPCServer import PerceptionServerThread
from perception.rosWrapper import ROSWrapper
# from opencood.visualization.vis_utils import color_encoding
from utils.sharedInfo import SharedInfo
from utils.perception_utils import get_lidar_pose_and_pcd_from_dataset, get_psa_from_obu, save_lidar_pose_and_pcd, \
    ros_pcd_to_numpy

import numpy as np


class PerceptionManager:
    def __init__(self, opt, cfg: AppConfig):
        self.ros_wrapper = None
        self.my_info = SharedInfo()
        self.running = False
        self.opt = opt
        self.cfg = cfg
        self.pcd_queue = deque(maxlen=10)
        self.perception_rpc_server = PerceptionServerThread(self.my_info)
        self.ros_thread = threading.Thread(target=self.__ros_start)

        if opt.show_vis:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(visible=True, window_name='Perception')
            vis_opt = self.vis.get_render_option()
            vis_opt.background_color = np.asarray([0, 0, 0])
            vis_opt.point_size = 1.0

    def start(self):
        self.running = True
        if not self.cfg.perception_debug:
            self.ros_thread.start()
        self.perception_rpc_server.start()

        self.__loop()

    def __ros_start(self):
        self.ros_wrapper = ROSWrapper(self.pcd_queue)
        self.ros_wrapper.start()
        logging.debug('roswrapper start')

    def __loop(self):
        loop_time = 0.1
        last_t = 0
        loop_index = 0
        while self.running:
            t = time.time()
            if t - last_t < loop_time:
                time.sleep(loop_time + last_t - t)
            last_t = time.time()

            self.__update_perception_info(loop_index)

            if self.opt.show_vis:
                pcd = self.my_info.get_pcd_copy()
                if pcd is not None:
                    self.__show_vis(pcd)

            loop_index += 1

    def __update_perception_info(self, loop_index):
        if self.cfg.perception_debug:
            perception_info = self.__get_info_from_dataset(loop_index)
        else:
            perception_info = self.__get_info(loop_index)
        if self.opt.save_pcd:
            save_lidar_pose_and_pcd(perception_info['lidar_pose'], perception_info['pcd'], file_name='001')
            self.close()

        self.my_info.update_perception_info_dict(perception_info)

    def __get_info_from_dataset(self, index):
        if self.cfg.static_asset_path.endswith(('.pcd', '.yaml', '.json', '.txt')):
            file_path = self.cfg.static_asset_path
        else:
            file_names = sorted([file_name for file_name in os.listdir(self.cfg.static_asset_path)
                                 if file_name.endswith(('.pcd', '.json', '.txt'))])
            file_path = os.path.join(self.cfg.static_asset_path, file_names[index % len(file_names)])

        lidar_pose, pcd = get_lidar_pose_and_pcd_from_dataset(file_path)
        perception_info = {'lidar_pose': lidar_pose,
                           'pcd': pcd}

        return perception_info

    def __get_info(self, loop_index):
        pcd = self.__retrieve_from_ros(loop_index)
        lidar_pose, speed, acceleration = get_psa_from_obu(self.cfg.obu_output_file_path)

        if self.cfg.perception_hea_debug:
            with open('./hea_debug.txt', 'r') as file:
                hea = float(file.readline())
                lidar_pose[5] = hea

        perception_info = {'lidar_pose': lidar_pose,
                           'pcd': pcd}

        print(lidar_pose)

        return perception_info

    def __retrieve_from_ros(self, index):
        # ori_pcd = self.pcd_queue.get()
        if len(self.pcd_queue) > 0:
            ros_pcd = self.pcd_queue.pop()

            pcd = ros_pcd_to_numpy(ros_pcd)
            logging.info(f'received valid pcd {len(pcd)}')
            return pcd

        return None

    def __show_vis(self, processed_pcd):
        self.vis.clear_geometries()
        pcd = o3d.geometry.PointCloud()

        # o3d use right-hand coordinate
        if self.cfg.perception_debug and self.cfg.perception_debug_data_from_OPV2V:
            processed_pcd[:, :1] = -processed_pcd[:, :1]

        pcd.points = o3d.utility.Vector3dVector(processed_pcd[:, :3])

        # origin_lidar_intcolor = color_encoding(processed_pcd[:, 2], mode='constant')
        # pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)

        self.vis.add_geometry(pcd)

        # 渲染场景
        self.vis.poll_events()
        self.vis.update_renderer()

        # self.vis.run()

    def close(self):
        self.running = False
        self.perception_rpc_server.close()
