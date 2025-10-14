import open3d as o3d
import numpy as np
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.append(parent_dir)

from src.utils.common import load_json
from test_utils import get_vis, get_pcd

if __name__ == '__main__':
    print(1)
    vis = get_vis()

    # pcd_file = r'D:\Documents\datasets\OPV2V\test_culver_city\2021_09_03_09_32_17\302\006220.pcd'
    pcd_file = '../pcds/25_07_09/198/json/302.json'
    # pcd_file = 'D:\\Documents\\datasets\\V2X4Real\\train_01\\testoutput_CAV_data_2022-03-15-10-09-50_0\\0\\000000.pcd'
    # pcd_file = r"D:\Documents\datasets\KiTTi\2011_09_26_drive_0113_sync\2011_09_26\2011_09_26_drive_0113_extract\2011_09_26\2011_09_26_drive_0113_extract\velodyne_points\data\0000000000.txt"
    pcd_np = get_pcd(pcd_file)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np[:, :3])
    vis.add_geometry(pcd)

    cam = vis.get_view_control().convert_to_pinhole_camera_parameters()

    # cam.extrinsic = np.array([[1, 0, 0, 0],  # 调整相机位置
    #                           [0, 1, 0, 0],
    #                           [0, 0, 1, 0],  # Z值增大=拉远相机
    #                           [0, 0, 0, 1]])
    extrinsic = np.array(cam.extrinsic)
    extrinsic[2, 3] -= 100
    cam.extrinsic = extrinsic
    vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

    # 渲染场景
    # vis.poll_events()
    # vis.update_renderer()
    vis.run()
