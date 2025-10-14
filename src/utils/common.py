import asyncio
from dataclasses import fields
import datetime
import hashlib
import json
import os
import re
import yaml
import base64
import logging
from typing import List, Optional, Any, Tuple, Type, TypeVar, Union
import traceback
import concurrent.futures
import appType
import numpy as np


def mstime() -> appType.timestamp_t:
    return appType.timestamp_t(datetime.datetime.now().timestamp() * 1000)

def panic():
    traceback.print_exc()
    assert False

def ms2s(s: int) -> float:
    return s / 1000

def load_json(file_path: str):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            d = json.load(f)
    except FileNotFoundError:
        d = None
    except json.JSONDecodeError as e:
        d = None
    return d

def load_yaml(file, opt=None):
    """
    Load yaml file and return a dictionary.

    Parameters
    ----------
    file : string
        yaml file path.

    opt : argparser
         Argparser.
    Returns
    -------
    param : dict
        A dictionary that contains defined parameters.
    """
    if opt and opt.model_dir:
        file = os.path.join(opt.model_dir, 'config.yaml')

    stream = open(file, 'r')
    loader = yaml.Loader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    param = yaml.load(stream, Loader=loader)
    if "yaml_parser" in param:
        param = eval(param["yaml_parser"])(param)

    return param



T = TypeVar('T')

def load_config(cls: Type[T], file_path: str) -> T:
    """
    从JSON文件加载配置到数据类
    
    参数:
        cls: 数据类类型
        file_path: JSON配置文件路径
        
    返回:
        数据类的实例，包含加载的配置
    """
    d = load_json(file_path)
    if d is None:
        config_data = {}
    else:
        config_data = d
    field_types = {f.name: f.type for f in fields(cls)}

    valid_fields = {k: v for k, v in config_data.items() if k in field_types}
    return cls(**valid_fields)

def server_assert(expr, info=''):
    assert expr, info

def server_logic_error(info: str):
    logging.warning(info)

def server_not_implemented(info: str):
    logging.warning(info)

def sync_to_async(sync_func):
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(pool, sync_func, *args, **kwargs)
        return result
    return wrapper

def read_binary_file(file_path: str, max_bytes: int = -1) -> bytes:
    """
    读取二进制文件的工具方法，支持全量读取或分块读取
    
    参数:
        file_path: 文件路径
        chunk_size: 分块读取时的块大小(字节)，为None时全量读取
        max_bytes: 最大读取字节数，超过时截断
    
    返回:
        bytes: 全量读取模式返回完整字节流
        list[bytes]: 分块读取模式返回字节块列表
    
    异常:
        FileNotFoundError: 文件不存在
        PermissionError: 无读取权限
        IsADirectoryError: 路径指向目录
        OSError: 其他文件操作错误
    """
    try:
        with open(file_path, 'rb') as file:
            # 全量读取模式
            data = file.read(max_bytes)
            return data
    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在: {file_path}")
    except PermissionError:
        raise PermissionError(f"无读取权限: {file_path}")
    except IsADirectoryError:
        raise IsADirectoryError(f"路径指向目录而非文件: {file_path}")
    except OSError as e:
        raise OSError(f"文件操作错误: {e}")
    
def string_to_32_hex(input_string):
    # 创建 md5 对象
    md5_hash = hashlib.md5()
    # 对输入的字符串进行编码，并更新 md5 对象
    md5_hash.update(input_string.encode('utf-8'))
    # 获取十六进制的哈希值
    return md5_hash.hexdigest()


def base64_encode(binary_data: bytes, encoding='utf-8') -> str:
    return base64.b64encode(binary_data).decode(encoding)


def base64_decode(str_data: str, encoding='utf-8') -> bytes:
    return base64.b64decode(str_data.encode(encoding))


def project_points_by_matrix_numpy(points, transformation_matrix):
    """
    基于变换矩阵将点投影到另一个坐标系。

    参数
    ----------
    points : np.ndarray
        3D点，形状为 (N, 3)

    transformation_matrix : np.ndarray
        变换矩阵，形状为 (4, 4)

    返回
    -------
    projected_points : np.ndarray
        投影后的点，形状为 (N, 3)
    """
    # 通过在最后一维填充1转换为齐次坐标
    # (N, 4)
    points_homogeneous = np.pad(points, ((0, 0), (0, 1)), mode='constant', constant_values=1)
    # (N, 4)
    projected_points = np.dot(points_homogeneous, transformation_matrix.T)
    return projected_points[:, :3]


def x_to_world(pose):
    """
    The transformation matrix from x-coordinate system to carla world system

    Parameters
    ----------
    pose : list
        [x, y, z, roll, yaw, pitch]

    Returns
    -------
    matrix : np.ndarray
        The transformation matrix.
    """
    x, y, z, roll, yaw, pitch = pose[:]

    # used for rotation matrix
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix

def x1_to_x2(x1, x2):
    """
    Transformation matrix from x1 to x2.

    Parameters
    ----------
    x1 : list
        The pose of x1 under world coordinates.
    x2 : list
        The pose of x2 under world coordinates.

    Returns
    -------
    transformation_matrix : np.ndarray
        The transformation matrix.

    """
    x1_to_world = x_to_world(x1)
    x2_to_world = x_to_world(x2)
    world_to_x2 = np.linalg.inv(x2_to_world)

    transformation_matrix = np.dot(world_to_x2, x1_to_world)
    return transformation_matrix


def euler_to_rotation_matrix(roll, pitch, yaw):
    """将欧拉角（弧度）转换为旋转矩阵（ZYX顺序）"""
    # 绕Z轴旋转（yaw）
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    # 绕Y轴旋转（pitch）
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    # 绕X轴旋转（roll）
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    # 总旋转矩阵：R = Rz * Ry * Rx（ZYX顺序）
    return np.dot(np.dot(Rz, Ry), Rx)

def get_world_transform(pose):
    """根据6维位姿（x,y,z,roll,pitch,yaw）生成世界坐标系变换矩阵"""
    x, y, z, roll, pitch, yaw = pose
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    # 齐次变换矩阵：[R | t; 0 0 0 1]
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = [x, y, z]
    return transform

def project_points_to_world(points, pose):
    """将自车坐标系下的点（N,3）投影到世界坐标系"""
    # 自车坐标系点转为齐次坐标（N,4）
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    # 获取变换矩阵
    transform = get_world_transform(pose)
    # 变换到世界坐标系
    world_points = np.dot(points_homogeneous, transform.T)[:, :3]
    return world_points

def calculate_confidence_map_overlap(
    map1: np.ndarray, pose1: np.ndarray,  # pose1 shape: (6,)
    map2: np.ndarray, pose2: np.ndarray,  # pose2 shape: (6,)
    grid_size: float = 0.4 ,  # 网格尺寸（米），假设z轴网格为0（俯视投影）
):
    """计算两个三维置信图的重叠率（投影到XY平面）"""
    h1, w1 = map1.shape
    h2, w2 = map2.shape

    # 生成自车1坐标系下的网格点（假设z=0，俯视投影）
    x1_grid, y1_grid = np.meshgrid(
        np.arange(-w1/2, w1/2) * grid_size,
        np.arange(-h1/2, h1/2) * grid_size
    )
    points1_local = np.stack([x1_grid.ravel(), y1_grid.ravel(), np.zeros_like(x1_grid.ravel())], axis=1)
    
    # 生成自车2坐标系下的网格点
    x2_grid, y2_grid = np.meshgrid(
        np.arange(-w2/2, w2/2) * grid_size,
        np.arange(-h2/2, h2/2) * grid_size
    )
    points2_local = np.stack([x2_grid.ravel(), y2_grid.ravel(), np.zeros_like(x2_grid.ravel())], axis=1)
    
    # 投影到世界坐标系
    points1_world = project_points_to_world(points1_local, pose1)
    points2_world = project_points_to_world(points2_local, pose2)
    
    # 提取XY平面坐标（忽略z轴）
    points1_xy = points1_world[:, :2]
    points2_xy = points2_world[:, :2]
    
    # 获取两个地图在XY平面的有效点（值为1的网格）
    valid1 = map1.ravel() == 1
    valid2 = map2.ravel() == 1
    valid_points1 = points1_xy[valid1]
    valid_points2 = points2_xy[valid2]
    
    # 计算重叠区域：使用二维平面的点集交集
    if valid_points1.size == 0 or valid_points2.size == 0:
        return 0.0
    
    # 基于KD树快速查找重叠点（适用于大规模数据）
    from scipy.spatial import KDTree
    tree1 = KDTree(valid_points1)
    distances, indices = tree1.query(valid_points2, k=1, distance_upper_bound=1e-6)  # 容差1e-6米
    overlap = np.sum(distances < 1e-6)
    
    # 计算总有效区域
    total = valid1.sum() + valid2.sum() - overlap
    if total == 0:
        return 0.0
    
    return overlap / total


def calculate_matrix_overlap(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
    if matrix1.shape != matrix2.shape:
        return -1.0

    overlap_elements = (matrix1 == matrix2)
    num_overlap = np.sum(overlap_elements)
    total_elements = matrix1.size
    overlap_ratio = num_overlap / total_elements

    return overlap_ratio
