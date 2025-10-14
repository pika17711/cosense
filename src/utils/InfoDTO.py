from dataclasses import dataclass
import pickle
from typing import Dict, Optional
from numpy.typing import NDArray
from utils.common import base64_encode, base64_decode
import appType
from appConfig import AppConfig
import numpy as np

@dataclass
class InfoDTO:
    type: int
    id: appType.id_t    # id
    lidar2world:        Optional[NDArray[np.float64]] # 雷达到世界的外参矩阵
    camera2world:       Optional[NDArray] # 相机到世界的外参矩阵
    camera_intrinsic:   Optional[NDArray] # 相机的内参矩阵
    feat:               Optional[Dict[str, NDArray]]  # 特征 {'spatial_feature': array, Optional['comm_mask': array]}
    ts_feat:            Optional[appType.timestamp_t] # 时间戳
    speed:              Optional[NDArray] # 速度
    ts_spd:           Optional[appType.timestamp_t] # 时间戳
    lidar_pos:          Optional[NDArray] # 位置
    ts_lidar_pos:       Optional[appType.timestamp_t] # 时间戳
    acc:                Optional[NDArray] # 加速度
    ts_acc:             Optional[appType.timestamp_t] # 时间戳
    pcd:                Optional[NDArray] # 点云 for vis and debug
    ts_pcd:             Optional[appType.timestamp_t] # 时间戳

class InfoDTOSerializer:
    @staticmethod
    def serialize(info_dto: InfoDTO, protocol: int = 4, compress: bool = False) -> bytes:
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
            data_dict = info_dto.__dict__
            
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
    def serialize_to_str(info_dto: InfoDTO, protocol: int = 4, compress: bool = False) -> str:
        """
        序列化 InfoDTO 对象为字符串

        参数:
            info_dto: 要序列化的 InfoDTO 对象
            protocol: pickle 协议版本 (默认 4，兼容 Python 3.4+)
            compress: 是否启用压缩 (需要安装 zlib)

        返回:
            str: 序列化后的字符串
        """
        binary_data = InfoDTOSerializer.serialize(info_dto, protocol, compress)

        try:
            str_data = base64_encode(binary_data)
            return str_data
        except Exception as e:
            print(f"序列化错误: {e}")
            return ""
    
    @staticmethod
    def deserialize(binary_data: bytes, decompress: bool = False) -> Optional[InfoDTO]:
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
            return InfoDTO(**data_dict)
        except Exception as e:
            print(f"反序列化错误: {e}")
            return None

    @staticmethod
    def deserialize_from_str(str_data: str, decompress: bool = False) -> Optional[InfoDTO]:
        """
        从字符串数据反序列化为 InfoDTO 对象

        参数:
            str_data: 要反序列化的字符串数据
            decompress: 是否需要先解压缩

        返回:
            InfoDTO: 反序列化后的对象
        """
        try:
            binary_data = base64_decode(str_data)
            return InfoDTOSerializer.deserialize(binary_data, decompress)
        except Exception as e:
            print(f"反序列化错误: {e}")