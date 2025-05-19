import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import time
import numpy as np
import open3d as o3d
from perception import perceptionRPCServer
from utils.sharedInfo import SharedInfo
from opencood.hypes_yaml.yaml_utils import load_yaml


def load_pcd(i):
    path = 'datasets/OPV2V/test_culver_city_part/2021_09_03_09_32_17/302/' + '{:06d}'.format(6220 + i * 2)

    yaml_load = load_yaml(path + '.yaml')
    pcd_load = o3d.io.read_point_cloud(path + '.pcd')

    pose = np.asarray(yaml_load['lidar_pose'])

    # 将Open3D的点云对象转换为NumPy数组
    xyz = np.asarray(pcd_load.points)

    # colors = np.zeros((len(xyz), 1))
    # pcd = np.hstack((xyz, colors))

    intensity = np.expand_dims(np.asarray(pcd_load.colors)[:, 0], -1)
    pcd = np.hstack((xyz, intensity))
    return pose, pcd


shared_info = SharedInfo()

service1_thread = perceptionRPCServer.PerceptionServerThread(shared_info)
service1_thread.setDaemon(True)
service1_thread.start()

try:
    # 保持主线程存活，防止程序退出
    while True:
        pose, pcd = load_pcd(int(time.time()) % 10)
        shared_info.update_pose(pose)
        print(pose)
        shared_info.update_pcd(pcd)
        time.sleep(1)
except KeyboardInterrupt:
    print("Server terminated.")
