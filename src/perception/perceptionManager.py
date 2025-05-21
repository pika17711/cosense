import logging
import time

import yaml.scanner

from perception.perceptionRPCServer import PerceptionServerThread
from utils.sharedInfo import SharedInfo
from utils.common import load_yaml
import open3d as o3d
import numpy as np


def load_pose_and_pcd(my_id, i):
    path = 'datasets/OPV2V/test_culver_city_part/2021_09_03_09_32_17/' + str(my_id) + '/' + '{:06d}'.format(6220 + i * 2)
    # path = 'D:\\Documents\\datasets\\OPV2V\\test_culver_city\\2021_09_03_09_32_17\\' + str(my_id) + '\\' + '{:06d}'.format(6220 + i * 2)

    try:
        yaml_load = load_yaml(path + '.yaml')
        pose = np.asarray(yaml_load['lidar_pose'])
    except FileNotFoundError as e:
        logging.error(e)
        pose = np.array([])
    except TypeError:
        logging.error(f'YAML file is empty or contains malformed content. File: \'{path}.yaml\'')
        pose = np.array([])
    except yaml.scanner.ScannerError:
        logging.error(f'YAML file contains malformed content. File: \'{path}.yaml\'')
        pose = np.array([])
    except KeyError as e:
        logging.error(f'Key {e} not found in YAML file. File: \'{path}.yaml\'')
        pose = np.array([])

    pcd_load = o3d.io.read_point_cloud(path + '.pcd')

    # 将Open3D的点云对象转换为NumPy数组
    xyz = np.asarray(pcd_load.points)
    intensity = np.expand_dims(np.asarray(pcd_load.colors)[:, 0], -1)
    pcd = np.hstack((xyz, intensity))
    return pose, pcd


class PerceptionManager:
    def __init__(self):
        self.my_info = SharedInfo()
        self.running = False

        self.perception_rpc_server = PerceptionServerThread(self.my_info)
        self.perception_rpc_server.start()

    def start(self):
        self.running = True
        self.__loop()

    def __loop(self):
        loop_time = 6
        last_t = 0

        while self.running:
            t = time.time()
            if t - last_t < loop_time:
                time.sleep(loop_time + last_t - t)
            last_t = time.time()

            pose, pcd = load_pose_and_pcd(302, (int(time.time() / loop_time)) % 60)
            self.my_info.update_perception_info(pose=pose, pcd=pcd)

    def close(self):
        self.running = False
        self.perception_rpc_server.close()
