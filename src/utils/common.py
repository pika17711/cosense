import asyncio
import datetime
import hashlib
import json
from typing import List, Optional, Any, Tuple, Union
import traceback
import concurrent.futures
from config import AppConfig
import numpy as np


def mstime() -> AppConfig.timestamp_t:
    return AppConfig.timestamp_t(datetime.datetime.now().timestamp() * 1000)

def panic():
    traceback.print_exc()
    assert False

def load_json(file_path: str):
    fp = open(file_path, "r")
    d = json.load(fp)
    return d

def server_assert(expr, info=''):
    assert expr, info

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