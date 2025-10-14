import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.append(parent_dir)

from src.utils.perception_utils import get_lidar_pose_and_pcd_from_dataset, save_lidar_pose_and_pcd

if __name__ == '__main__':
    path = r"D:\Documents\datasets\OPV2V\test_culver_city_part\2021_09_03_09_32_17\302\006220.pcd"
    lidar_pose, pcd = get_lidar_pose_and_pcd_from_dataset(path)
    save_lidar_pose_and_pcd(lidar_pose, pcd)
