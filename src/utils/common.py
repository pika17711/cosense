import asyncio
import datetime
from typing import Optional, Any, Tuple

import numpy as np


def mstime() -> float:
    return datetime.datetime.now().timestamp()

import numpy as np
from scipy.ndimage import zoom

def transform_feature(feature, target_h, target_w):
    """
    调整特征的高度和宽度以匹配给定的目标尺寸。

    参数:
        feature (np.ndarray): 输入特征，形状为 (C, H, W)。
        target_h (int): 目标高度。
        target_w (int): 目标宽度。

    返回:
        np.ndarray: 调整后的特征，形状为 (C, target_h, target_w)。
    """
    C, H, W = feature.shape
    zoom_factors = (target_h / H, target_w / W)
    resized_feature = zoom(feature, zoom_factors)
    return resized_feature

def transform_communication_mask(mask, target_h, target_w):
    """
    调整通信掩码的高度和宽度以匹配给定的目标尺寸。

    参数:
        mask (np.ndarray): 输入通信掩码，形状为 (H, W)。
        target_h (int): 目标高度。
        target_w (int): 目标宽度。

    返回:
        np.ndarray: 调整后的通信掩码，形状为 (target_h, target_w)。
    """
    _, H, W = mask.shape
    zoom_factors = (target_h / H, target_w / W)
    resized_mask = zoom(mask, zoom_factors)
    return resized_mask


def calculate_overlap_ratio(mask1, mask2):
    """
    计算两个通信掩码的重叠率
    :param mask1: 第一个通信掩码，形状为 (H, W)
    :param mask2: 第二个通信掩码，形状为 (H, W)
    :return: 重叠率
    """
    # 计算交集
    intersection = np.logical_and(mask1, mask2)
    intersection_count = np.sum(intersection)

    # 计算并集
    union = np.logical_or(mask1, mask2)
    union_count = np.sum(union)

    # 计算重叠率
    if union_count == 0:
        return 0
    overlap_ratio = intersection_count / union_count
    return overlap_ratio