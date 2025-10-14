
from dataclasses import dataclass
from enum import IntEnum, auto
import logging
import pickle
from typing import Optional, Union
import appType
from numpy.typing import NDArray

import numpy as np
from utils.common import calculate_confidence_map_overlap, x1_to_x2, calculate_matrix_overlap
from utils.common import project_points_by_matrix_numpy

class CoopMapType(IntEnum):
    DEBUG = auto()
    Empty = auto()
    CommMask = auto()
    RequestMap = auto()
    Unknown = auto()

class CoopMap:
    def __init__(self, oid: appType.id_t, type: CoopMapType, map: Optional[NDArray], lidar_pose: Optional[NDArray]) -> None:
        self.oid = oid
        self.type = type
        self.map = map
        self.lidar_pose = lidar_pose

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
            # data_dict = coopmap.__dict__
            data_dict = coopmap.__dict__.copy()
            if isinstance(data_dict['map'], np.ndarray):
                np_map = data_dict['map'].astype(np.int8)
                packed_map = np.packbits(np_map, axis=-1)
                binary_map = packed_map.tobytes()

                data_dict['map'] = binary_map

            binary_data = pickle.dumps(data_dict, protocol=protocol)
            
            if compress:
                import zlib
                binary_data = zlib.compress(binary_data)
                
            return binary_data
        except Exception as e:
            print(f"序列化错误: {e}")
            return b''

    @staticmethod
    def serialize_only_map_and_lidar_pose(coopmap: 'CoopMap', compress: bool = False) -> bytes:
        try:
            data_dict = coopmap.__dict__.copy()

            np_map = data_dict['map'].astype(np.int8)
            packed_map = np.packbits(np_map, axis=-1)
            binary_map = packed_map.tobytes()

            binary_lidar_pose = data_dict['lidar_pose'].tobytes()

            binary_data = binary_map + binary_lidar_pose

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
            if decompress:
                import zlib
                binary_data = zlib.decompress(binary_data)
                
            data_dict = pickle.loads(binary_data)

            if data_dict.get('map') is not None:
                binary_map = data_dict['map']
                packed_map = np.frombuffer(binary_map, dtype=np.uint8).reshape(1, 1, 48, -1)
                np_map = np.unpackbits(packed_map, axis=-1).astype(np.float32)

                data_dict['map'] = np_map
            
            return CoopMap(**data_dict)
        except Exception as e:
            logging.error(f"反序列化错误: {e}")
            return None

    @staticmethod
    def deserialize_from_only_map_and_lidar_pose(binary_data: bytes, decompress: bool = False) -> Optional['CoopMap']:
        try:
            if decompress:
                import zlib
                binary_data = zlib.decompress(binary_data)

            binary_map = binary_data[:1056]     # 1056 = 48 * 176 / 8
            binary_lidar_pose = binary_data[1056:]

            packed_map = np.frombuffer(binary_map, dtype=np.uint8).reshape(1, 1, 48, -1)
            np_map = np.unpackbits(packed_map, axis=-1).astype(np.float32)

            lidar_pose = np.frombuffer(binary_lidar_pose, dtype=np.float64)

            return CoopMap('', CoopMapType.Unknown, np_map, lidar_pose)

        except Exception as e:
            logging.error(f"反序列化错误: {e}")
            return None

    @staticmethod
    def calculate_overlap_ratio(map1: 'CoopMap', map2: 'CoopMap', debug=False) -> float:
        if debug or map1.type == CoopMapType.DEBUG or map2.type == CoopMapType.DEBUG:
            return 1.0
        # if map1.type != CoopMapType.CommMask or map2.type != CoopMapType.CommMask:
        #     return 0.0
        # return calculate_confidence_map_overlap(map1.map, map1.lidar_pose, map2.map, map2.lidar_pose)

        if (map1.type == CoopMapType.RequestMap and map2.type == CoopMapType.CommMask) or \
           (map2.type == CoopMapType.RequestMap and map1.type == CoopMapType.CommMask):
            return calculate_matrix_overlap(map1.map, map2.map)

        return 0.0
