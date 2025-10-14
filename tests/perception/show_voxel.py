import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
import numpy as np
import open3d as o3d
from test_utils import get_vis, get_pcd
from tests.detection.inference_test import get_shared_info
from utils.detection_utils import pcd_to_voxel

if __name__ == '__main__':
    vis = get_vis()

    # pcd_file = r'D:\Documents\datasets\OPV2V\test_culver_city\2021_09_03_09_32_17\302\006220.pcd'
    # pcd_file = r"D:\Documents\datasets\KiTTi\2011_09_26_drive_0113_sync\2011_09_26\2011_09_26_drive_0113_extract\2011_09_26\2011_09_26_drive_0113_extract\velodyne_points\data\0000000000.txt"
    pcd_file = r'../pcds/25_07_09/199/json/301.json'

    pcd_np = get_pcd(pcd_file)
    pcd_np[:, 3] = 1
    # step = 1
    # for i in range(0, pcd_np.shape[0], step):
    #     print(i)
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(pcd_np[i:i+step, :3].reshape(step, 3))
    #     vis.add_geometry(pcd)
    #     vis.poll_events()
    #     vis.update_renderer()

    shared_info = get_shared_info()

    processed_pcd, voxel = pcd_to_voxel(pcd_np, shared_info)

    grid_map = np.zeros((192, 704), dtype=bool)

    x_coords = voxel['voxel_coords'][:, 1].astype(int)
    y_coords = voxel['voxel_coords'][:, 2].astype(int)
    grid_map[x_coords, y_coords] = True

    # 创建可视化
    plt.figure(figsize=(14, 6))  # 设置图像尺寸 (宽度, 高度)

    # 创建自定义颜色映射: False→白色, True→绿色
    cmap = ListedColormap(['white', 'green'])

    # 绘制栅格地图
    # 注意: 使用origin='lower'确保y轴从下到上增加
    plt.imshow(grid_map, cmap=cmap, origin='lower',
               extent=[-0.5, 703.5, -0.5, 191.5],
               aspect='auto', interpolation='nearest')

    # 计算中心点坐标 (704列的中心是351.5, 192行的中心是95.5)
    center_x = 351.5
    center_y = 95.5

    # 在中心位置添加红色圆点表示智能车
    center_point = Circle((center_x, center_y), radius=5, color='red', zorder=10)
    plt.gca().add_patch(center_point)

    # 添加标题和坐标轴标签
    plt.title('Smart Car Perception Visualization', fontsize=16)
    plt.xlabel('X Coordinate (0-703)', fontsize=12)
    plt.ylabel('Y Coordinate (0-191)', fontsize=12)

    # 添加网格线以便更好地观察位置
    plt.grid(True, color='gray', linestyle='--', linewidth=0.3, alpha=0.5)

    # 添加比例尺
    plt.text(10, 180, 'Each square = 1 unit', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8))

    # 添加图例说明
    plt.text(500, 180, 'Green: Detected Objects', color='green', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8))
    plt.text(500, 170, 'Red: Smart Car Position', color='red', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    # plt.show()



    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np[:, :3])
    vis.add_geometry(pcd)

    # 渲染场景
    # vis.poll_events()
    # vis.update_renderer()
    # vis.run()
    plt.show()
