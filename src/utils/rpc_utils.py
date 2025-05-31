import numpy as np

from rpc import Service_pb2


def np_to_protobuf(array_np):
    return Service_pb2.NDArray(data=array_np.tobytes(),
                               dtype=str(array_np.dtype),
                               shape=list(array_np.shape))


def protobuf_to_np(array_protobuf):
    return np.frombuffer(array_protobuf.data, dtype=array_protobuf.dtype).reshape(array_protobuf.shape)


def protobuf_to_dict(dict_protobuf):
    dict_python = {}
    for cav_id, info_protobuf in dict_protobuf.items():
        info_python = {}
        for field in info_protobuf.DESCRIPTOR.fields:
            key = field.name
            value = getattr(info_protobuf, key)
            if isinstance(value, Service_pb2.NDArray):
                value = protobuf_to_np(value)
            info_python[key] = value
        dict_python[cav_id] = info_python
    return dict_python
