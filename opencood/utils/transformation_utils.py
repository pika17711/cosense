# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Transformation utils
"""

import math
import numpy as np
from pyproj import Transformer, Geod


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


def get_utm_epsg(lon, lat):
    """获取给定经纬度的UTM EPSG代码"""
    zone = math.floor((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def gps_to_utm(pose, target_epsg=None):
    """
    将GPS坐标(经度, 纬度, 海拔)转换为UTM坐标，并调整航向角

    参数:
        pose: [lon, lat, alt, 0, 0, hea]
            lon: 经度 (度)
            lat: 纬度 (度)
            alt: 海拔高度 (米)
            hea: 航向角 (度，正北顺时针)

    返回:
        [x, y, z, roll, pitch, yaw] 格式的位姿
            x: 东移值 (米)
            y: 北移值 (米)
            z: 海拔高度 (米)
    """
    lon, lat, alt, _, _, hea = pose

    # 如果没有指定目标EPSG，则自动确定
    if target_epsg is None:
        target_epsg = get_utm_epsg(lon, lat)

    # 创建WGS84到UTM的转换器
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{target_epsg}", always_xy=True)

    # 转换坐标
    x, y = transformer.transform(lon, lat)
    z = alt

    # 转换航向角：正北顺时针 → 正东逆时针
    # 航向角0°(北) → 90°, 90°(东) → 0°, 180°(南) → -90°
    yaw = math.radians(90 - hea)

    return [x, y, z, 0, 0, yaw]  # roll和pitch设为0


def gps_to_enu(ref_gps, target_gps):
    """
    将目标GPS坐标转换为相对于参考点的ENU坐标（东-北-天）

    参数:
        ref_gps: 参考点GPS位姿 [lon_ref, lat_ref, alt_ref, 0, 0, hea_ref]
        target_gps: 目标点GPS位姿 [lon, lat, alt, 0, 0, hea]

    返回:
        enu_pose: [e, n, u, 0, 0, yaw] 格式的位姿
    """
    # 解析参考点和目标点GPS数据
    lon_ref, lat_ref, alt_ref, _, _, hea_ref = ref_gps
    lon, lat, alt, _, _, hea = target_gps

    # 创建大地测量对象
    geod = Geod(ellps='WGS84')

    # 计算相对方位角、距离和高差
    az, _, dist = geod.inv(lon_ref, lat_ref, lon, lat)
    du = alt - alt_ref

    # 将距离分解为东向和北向分量
    # 方位角az是从北向顺时针的角度（与航向角定义一致）
    azimuth_rad = math.radians(az)
    e = dist * math.sin(azimuth_rad)  # 东向分量
    n = dist * math.cos(azimuth_rad)  # 北向分量

    # 计算绝对航向角（在ENU坐标系中）
    # 航向角0°(北) → 90°(东), 90°(东) → 0°, 180°(南) → -90°(西)
    yaw = math.radians(90 - hea)

    return [e, n, du, 0, 0, yaw]


def pose_to_matrix(pose):
    """
    将位姿[x, y, z, roll, pitch, yaw]转换为4x4变换矩阵

    参数:
        pose: 位姿列表 [x, y, z, roll, pitch, yaw]

    返回:
        4x4齐次变换矩阵
    """
    x, y, z, roll, pitch, yaw = pose

    # 创建旋转矩阵 (仅考虑yaw，忽略roll和pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    # 绕Z轴旋转 (yaw)
    rotation_matrix = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ])

    # 创建4x4变换矩阵
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[0, 3] = x
    transformation_matrix[1, 3] = y
    transformation_matrix[2, 3] = z

    return transformation_matrix


def gps_to_utm_transformation(x1_gps, x2_gps):
    """
    计算从x1坐标系到x2坐标系的变换矩阵

    参数:
        x1_gps: x1的GPS位姿 [lon1, lat1, alt1, 0, 0, hea1]
        x2_gps: x2的GPS位姿 [lon2, lat2, alt2, 0, 0, hea2]

    返回:
        4x4变换矩阵 (numpy数组)
    """
    # 解析GPS数据
    lon1, lat1, alt1, _, _, hea1 = x1_gps
    lon2, lat2, alt2, _, _, hea2 = x2_gps

    # 确定参考UTM分区（以x1所在分区为参考）
    ref_epsg = get_utm_epsg(lon1, lat1)

    # 转换为UTM坐标系
    x1_pose = gps_to_utm(x1_gps, target_epsg=ref_epsg)
    x2_pose = gps_to_utm(x2_gps, target_epsg=ref_epsg)

    # 转换为变换矩阵
    x1_to_world = pose_to_matrix(x1_pose)
    x2_to_world = pose_to_matrix(x2_pose)

    # 计算世界坐标系到x2坐标系的逆变换
    world_to_x2 = np.linalg.inv(x2_to_world)

    # 计算从x1到x2的变换
    transformation_matrix = np.dot(world_to_x2, x1_to_world)

    return transformation_matrix


def gps_to_enu_transformation(x1_gps, x2_gps):
    """
    计算从x1坐标系到x2坐标系的变换矩阵（基于ENU坐标系）

    参数:
        x1_gps: x1的GPS位姿 [lon1, lat1, alt1, 0, 0, hea1]
        x2_gps: x2的GPS位姿 [lon2, lat2, alt2, 0, 0, hea2]

    返回:
        4x4变换矩阵 (numpy数组)
    """
    # 以x1为参考点建立ENU坐标系
    # x1在ENU坐标系中的位姿是原点
    # 计算绝对航向角（在ENU坐标系中）
    # 航向角0°(北) → 90°(东), 90°(东) → 0°, 180°(南) → -90°(西)
    x1_pose = [0, 0, 0, 0, 0, math.radians(90 - x1_gps[5])]  # (e, n, u) = (0,0,0), yaw=90-hea1

    # 计算x2相对于x1的ENU位姿
    x2_pose = gps_to_enu(x1_gps, x2_gps)

    # 转换为变换矩阵
    x1_to_world = pose_to_matrix(x1_pose)
    x2_to_world = pose_to_matrix(x2_pose)

    # 计算世界坐标系(ENU)到x2坐标系的逆变换
    world_to_x2 = np.linalg.inv(x2_to_world)

    # 计算从x1到x2的变换
    transformation_matrix = np.dot(world_to_x2, x1_to_world)

    return transformation_matrix


def dist_to_continuous(p_dist, displacement_dist, res, downsample_rate):
    """
    Convert points discretized format to continuous space for BEV representation.
    Parameters
    ----------
    p_dist : numpy.array
        Points in discretized coorindates.

    displacement_dist : numpy.array
        Discretized coordinates of bottom left origin.

    res : float
        Discretization resolution.

    downsample_rate : int
        Dowmsamping rate.

    Returns
    -------
    p_continuous : numpy.array
        Points in continuous coorindates.

    """
    p_dist = np.copy(p_dist)
    p_dist = p_dist + displacement_dist
    p_continuous = p_dist * res * downsample_rate
    return p_continuous
