import open3d as o3d
import numpy as np
from opencood.utils.transformation_utils import x1_to_x2, x_to_world, gps_to_enu_transformation, gps_to_utm_transformation
from opencood.utils import box_utils
from test_utils import get_pcd, get_vis

if __name__ == '__main__':
    vis = get_vis()
    vis.get_render_option().point_size = 1.0

    pcd_file = r'D:\Documents\datasets\OPV2V\test_culver_city\2021_09_03_09_32_17\302\006220.pcd'
    pcd_np = get_pcd(pcd_file)

    pcd_np[:, :1] = - pcd_np[:, :1]

    x1 = [0, 0, 0, 0, 0, 0]
    x2 = [0, 0, 0, 0, 0, 90]

    pcd_ori = o3d.geometry.PointCloud()
    pcd_ori.points = o3d.utility.Vector3dVector(pcd_np[:, :3])
    red_color = np.array([1.0, 0, 0])  # RGB值范围[0,1]
    pcd_ori.colors = o3d.utility.Vector3dVector(np.tile(red_color, (len(pcd_ori.points), 1)))  # 复制颜色到所有点
    vis.add_geometry(pcd_ori)

    # transformation_matrix = x1_to_x2(x1, x2)
    transformation_matrix_enu = gps_to_enu_transformation(x1, x2)
    transformation_matrix_utm = gps_to_utm_transformation(x1, x2)

    pcd_np_projected_to_x2_enu = box_utils.project_points_by_matrix_torch(pcd_np[:, :3], transformation_matrix_enu)

    pcd_projected_enu = o3d.geometry.PointCloud()
    pcd_projected_enu.points = o3d.utility.Vector3dVector(pcd_np_projected_to_x2_enu[:, :3])
    blue_color = np.array([0, 0, 1.0])  # RGB值范围[0,1]
    pcd_projected_enu.colors = o3d.utility.Vector3dVector(np.tile(blue_color, (len(pcd_projected_enu.points), 1)))  # 复制颜色到所有点
    vis.add_geometry(pcd_projected_enu)

    pcd_np_projected_to_x2_utm = box_utils.project_points_by_matrix_torch(pcd_np[:, :3], transformation_matrix_utm)

    pcd_projected_utm = o3d.geometry.PointCloud()
    pcd_projected_utm.points = o3d.utility.Vector3dVector(pcd_np_projected_to_x2_utm[:, :3])
    green_color = np.array([0, 1.0, 0])  # RGB值范围[0,1]
    pcd_projected_utm.colors = o3d.utility.Vector3dVector(np.tile(green_color, (len(pcd_projected_utm.points), 1)))  # 复制颜色到所有点
    vis.add_geometry(pcd_projected_utm)

    vis.run()
