import math

import numpy as np
import open3d as o3d

from opencood.utils import box_utils
from utils.perception_utils import get_lidar_pose_and_pcd_from_dataset
from opencood.utils.transformation_utils import gps_to_utm_transformation, gps_to_enu_transformation, gps_to_enu


def cal_distance(lidar_pose1, lidar_pose2):
    e, n, du, _, _, _ = gps_to_enu(lidar_pose1, lidar_pose2)
    print(f'e: {e}m')
    print(f'n: {n}m')
    print(f'dis: {math.sqrt(e * e + n * n)}m')



if __name__ == '__main__':
    root_path = '../pcds/25_07_09/'
    cav1_id = '198'         # 路
    cav2_id = '199'         # 车
    pcd_id = '302'

    pcd1_path = root_path + cav1_id + '/json/' + pcd_id + '.json'
    pcd2_path = root_path + cav2_id + '/json/' + pcd_id + '.json'

    lidar_pose1, pcd_np1 = get_lidar_pose_and_pcd_from_dataset(pcd1_path)
    lidar_pose2, pcd_np2 = get_lidar_pose_and_pcd_from_dataset(pcd2_path)

    lidar_pose2[2] = lidar_pose1[2]

    cal_distance(lidar_pose1, lidar_pose2)

    transformation_matrix_2 = gps_to_utm_transformation(lidar_pose2, lidar_pose1)
    transformation_matrix_3 = gps_to_enu_transformation(lidar_pose2, lidar_pose1)
    print(transformation_matrix_2)
    print(transformation_matrix_3)

    proed_pcd_np2 = pcd_np2.copy()
    proed_pcd_np2[:, :3] = box_utils.project_points_by_matrix_torch(proed_pcd_np2[:, :3], transformation_matrix_2)

    proed_pcd_np3 = pcd_np2.copy()
    proed_pcd_np3[:, :3] = box_utils.project_points_by_matrix_torch(proed_pcd_np3[:, :3], transformation_matrix_3)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis_opt = vis.get_render_option()
    vis_opt.background_color = np.asarray([0, 0, 0])
    vis_opt.point_size = 2.0

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])  # 坐标系
    vis.add_geometry(axis)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pcd_np1[:, :3])
    red_color = np.array([1.0, 0, 0])  # RGB值范围[0,1]
    pcd1.colors = o3d.utility.Vector3dVector(np.tile(red_color, (len(pcd1.points), 1)))  # 复制颜色到所有点
    vis.add_geometry(pcd1)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(proed_pcd_np2[:, :3])
    blue_color = np.array([0, 0, 1.0])  # RGB值范围[0,1]
    pcd2.colors = o3d.utility.Vector3dVector(np.tile(blue_color, (len(pcd2.points), 1)))  # 复制颜色到所有点
    vis.add_geometry(pcd2)

    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(proed_pcd_np3[:, :3])
    blue_color = np.array([0, 1.0, 0])  # RGB值范围[0,1]
    pcd3.colors = o3d.utility.Vector3dVector(np.tile(blue_color, (len(pcd3.points), 1)))  # 复制颜色到所有点
    vis.add_geometry(pcd3)

    vis.run()
