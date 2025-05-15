
from dataclasses import dataclass
from enum import IntEnum, auto
import pickle
from typing import Optional, Union

import numpy as np
from utils.common import calculate_confidence_map_overlap, x1_to_x2
from utils.common import project_points_by_matrix_numpy

class CoopMapType(IntEnum):
    DEBUG = auto()
    WHERE2COMM = auto()

class CoopMap:
    def __init__(self, oid, type: CoopMapType, mp: Optional[np.ndarray], pose: Optional[np.ndarray]) -> None:
        self.oid = oid
        self.type = type
        self.map = mp
        self.lidar_pose = pose

    @staticmethod
    def serialize(coopmap: 'CoopMap', protocol: int = 4, compress: bool = False) -> bytes:
        """
        序列化 InfoDTO 对象为二进制数据
        
        参数:
            info_dto: 要序列化的 InfoDTO 对象
            protocol: pickle 协议版本 (默认 4，兼容 Python 3.4+)
            compress: 是否启用压缩 (需要安装 zlib)
            
        返回:
            bytes: 序列化后的二进制数据
        """
        try:
            # 将对象转换为字典
            data_dict = coopmap.__dict__
            
            # 序列化
            binary_data = pickle.dumps(data_dict, protocol=protocol)
            
            # 压缩 (可选)
            if compress:
                import zlib
                binary_data = zlib.compress(binary_data)
                
            return binary_data
        except Exception as e:
            print(f"序列化错误: {e}")
            return b''

    @staticmethod
    def deserialize(binary_data: bytes, decompress: bool = False) -> Optional['CoopMap']:
        """
        从二进制数据反序列化为 InfoDTO 对象
        
        参数:
            binary_data: 要反序列化的二进制数据
            decompress: 是否需要先解压缩
            
        返回:
            InfoDTO: 反序列化后的对象
        """
        try:
            # 解压缩 (可选)
            if decompress:
                import zlib
                binary_data = zlib.decompress(binary_data)
                
            # 反序列化
            data_dict = pickle.loads(binary_data)
            
            # 重建 InfoDTO 对象
            return CoopMap(**data_dict)
        except Exception as e:
            print(f"反序列化错误: {e}")
            return None

    @staticmethod
    def calculate_overlap_ratio(map1: 'CoopMap', map2: 'CoopMap') -> float:
        if map1.type == CoopMapType.DEBUG or map2.type == CoopMapType.DEBUG:
            return 1.0
        return calculate_confidence_map_overlap(map1.map, map1.lidar_pose, map2.map, map2.lidar_pose)