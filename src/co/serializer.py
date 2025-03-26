import numpy as np
from sense_pb2 import NDArray, DataItem, ParamData, ParamValue

class Serializer:
    @staticmethod
    def serialize_ndarray(array: np.ndarray) -> NDArray:
        """序列化NumPy数组"""
        return NDArray(
            data=array.tobytes(),
            shape=array.shape,
            dtype=array.dtype.str
        )

    @staticmethod
    def deserialize_ndarray(proto: NDArray) -> np.ndarray:
        """反序列化NumPy数组"""
        return np.frombuffer(
            proto.data, 
            dtype=np.dtype(proto.dtype)
        ).reshape(proto.shape)

    @classmethod
    def serialize_param(cls, param: dict) -> ParamData:
        """序列化嵌套参数字典到 ParamData 消息"""
        param_data = ParamData()
        
        for key, value in param.items():
            if isinstance(value, np.ndarray):
                # 处理数组类型
                param_data.values[key].array_value.CopyFrom(
                    cls.serialize_ndarray(value)
                )
            elif isinstance(value, int):
                # 处理整型
                param_data.values[key].int_value = value
            else:
                raise TypeError(f"Unsupported parameter type: {type(value)}")
        
        return param_data

    @classmethod
    def serialize_data_item(cls, item: dict) -> DataItem:
        """序列化单个数据项"""
        return DataItem(
            lidar_np=cls.serialize_ndarray(item['lidar_np']),
            param=cls.serialize_param(item['param'])
        )

    @classmethod
    def serialize_request(cls, data: dict) -> dict:
        """构建完整请求"""
        return {k: cls.serialize_data_item(v) for k, v in data.items()}