import numpy as np
import open3d as o3d
from src.utils.common import load_yaml, load_json


def grid_map(k, interval=10):
    temp_0 = np.arange(-k, k + interval, interval).reshape((-1, 1))
    temp_1 = np.array([-k for i in range(len(temp_0))]).reshape((-1, 1))
    temp_2 = np.array([k for i in range(len(temp_0))]).reshape((-1, 1))

    loc_left = np.concatenate((temp_1, temp_0), 1)
    loc_right = np.concatenate((temp_2, temp_0), 1)
    loc_lower = np.concatenate((temp_0, temp_1), 1)
    loc_upper = np.concatenate((temp_0, temp_2), 1)
    # Vertical and horizontal
    lineset_vertical = o3d.geometry.LineSet()
    points_vertical = np.concatenate((loc_lower, loc_upper), 0)
    points_vertical = np.concatenate((points_vertical, np.zeros((len(points_vertical), 1))), 1)
    lines_box_vertical = []
    for i in range(len(points_vertical) // 2):
        lines_box_vertical.append([i, i + len(loc_lower)])
    colors = np.array([[0.2, 0.2, 0.2] for j in range(len(lines_box_vertical))])
    lineset_vertical.points = o3d.utility.Vector3dVector(points_vertical)
    lineset_vertical.lines = o3d.utility.Vector2iVector(np.array(lines_box_vertical))
    lineset_vertical.colors = o3d.utility.Vector3dVector(colors)

    lineset_horizontal = o3d.geometry.LineSet()
    points_horizontal = np.concatenate((loc_left, loc_right), 0)
    points_horizontal = np.concatenate((points_horizontal, np.zeros((len(points_horizontal), 1))), 1)
    lines_box_horizontal = []
    for j in range(len(points_horizontal) // 2):
        lines_box_horizontal.append([j, j + len(loc_lower)])
    colors = np.array([[0.2, 0.2, 0.2] for j in range(len(lines_box_horizontal))])
    lineset_horizontal.points = o3d.utility.Vector3dVector(points_horizontal)
    lineset_horizontal.lines = o3d.utility.Vector2iVector(np.array(lines_box_horizontal))
    lineset_horizontal.colors = o3d.utility.Vector3dVector(colors)
    return [lineset_vertical, lineset_horizontal]


def grid_map_1(x_range=(-140.8, 140.8), y_range=(-38.4, 38.4), z=-3.0, grid_size=3.2):
    """
        在 z 平面绘制一个网格。

        Args:
            x_range (tuple): (x_min, x_max)
            y_range (tuple): (y_min, y_max)
            z(float): 网格高度
            grid_size (float): 每个网格的边长
        """
    lines = []
    points = []

    x_min, x_max = x_range
    y_min, y_max = y_range

    # 垂直线
    x_coords = np.arange(x_min, x_max + grid_size, grid_size)
    for x in x_coords:
        p1 = [x, y_min, z]
        p2 = [x, y_max, z]
        points.extend([p1, p2])
        lines.append([len(points) - 2, len(points) - 1])

    # 水平线
    y_coords = np.arange(y_min, y_max + grid_size, grid_size)
    for y in y_coords:
        p1 = [x_min, y, z]
        p2 = [x_max, y, z]
        points.extend([p1, p2])
        lines.append([len(points) - 2, len(points) - 1])

    colors = [[0.5, 0.5, 0.5] for _ in range(len(lines))]  # 灰色线条

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def get_vis():
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis_opt = vis.get_render_option()
    vis_opt.background_color = np.asarray([0, 0, 0])
    vis_opt.point_size = 2.0

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])  # 坐标系
    vis.add_geometry(axis)

    grid_map = grid_map_1()
    vis.add_geometry(grid_map)

    return vis


def get_pcd(pcd_file):
    pcd_np = []
    if pcd_file.endswith('.pcd'):
        with open(pcd_file, 'r') as file:
            data = False

            for line in file:
                if not data:
                    if line.startswith('DATA'):
                        print(line)
                        data = True
                    continue

                splits = line.split(' ')
                pcd_np.append(np.array([float(num) for num in splits], dtype=np.float32))
        pcd_np = np.vstack(pcd_np)

        # pcd_load = o3d.io.read_point_cloud(pcd_file)
        #
        # # 将Open3D的点云对象转换为NumPy数组
        # xyz = np.asarray(pcd_load.points)
        # colors = np.asarray(pcd_load.colors)
        # intensity = np.expand_dims(np.asarray(pcd_load.colors)[:, 0], -1)
        # pcd_np = np.hstack((xyz, intensity))
    elif pcd_file.endswith('.txt'):
        with open(pcd_file, 'r') as file:
            for line in file:
                splits = line.split(' ')
                pcd_np.append(np.array([float(num) for num in splits]))
        pcd_np = np.vstack(pcd_np)
    elif pcd_file.endswith('.json'):
        json_load = load_json(pcd_file)

        pcd_np = np.array(json_load['pcd']) if 'pcd' in json_load else None

    return pcd_np
