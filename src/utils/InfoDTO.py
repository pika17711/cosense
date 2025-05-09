from dataclasses import dataclass
import pickle
from typing import Dict
from config import AppConfig
import numpy as np

@dataclass
class InfoDTO:
    type: int
    id: AppConfig.id_t # id
    lidar2world: np.ndarray # 雷达到世界的外参矩阵
    camera2world: np.ndarray # 相机到世界的外参矩阵
    camera_intrinsic: np.ndarray # 相机的内参矩阵

    feat: Dict[str, np.ndarray] # 特征 {'voxel_features': array, 'voxel_coords': array, 'voxel_num_points': array}
    ts_feat: int # 时间戳
    speed: np.ndarray # 速度
    ts_speed: int # 时间戳
    pos: np.ndarray # 位置
    ts_pos: int # 时间戳
    acc: np.ndarray # 加速度
    ts_acc: int # 时间戳

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
    def deserialize(binary_data: bytes, decompress: bool = False) -> InfoDTO:
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