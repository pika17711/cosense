import threading

import grpc
from concurrent import futures
from rpc import Service_pb2
from rpc import Service_pb2_grpc
import time
import numpy as np


class PerceptionService(Service_pb2_grpc.PerceptionServiceServicer):  # 感知子系统的Service类
    def GetMyFeature(self, request, context):  # 感知子系统向其他进程提供“获取自车特征”的服务

        # ###################################################需要真实数据来源
        timestamp = int(time.time())                        # 时间戳
        my_feature = {                                      # 自车特征
            'voxel_features': np.array([101, 102, 103]),
            'voxel_coords': np.array([104, 105, 106]),
            'voxel_num_points': np.array([107, 108, 109])
        }
        # ###################################################

        return Service_pb2.MyFeature(  # 序列化并返回自车特征
            timestamp=timestamp,
            my_feature=Service_pb2.Feature(
                voxel_features=Service_pb2.NdArray(
                    data=my_feature['voxel_features'].tobytes(),
                    dtype=str(my_feature['voxel_features'].dtype),
                    shape=list(my_feature['voxel_features'].shape)
                ),
                voxel_coords=Service_pb2.NdArray(
                    data=my_feature['voxel_coords'].tobytes(),
                    dtype=str(my_feature['voxel_coords'].dtype),
                    shape=list(my_feature['voxel_coords'].shape)
                ),
                voxel_num_points=Service_pb2.NdArray(
                    data=my_feature['voxel_num_points'].tobytes(),
                    dtype=str(my_feature['voxel_num_points'].dtype),
                    shape=list(my_feature['voxel_num_points'].shape)
                )
            )
        )

    def GetMyConfMap(self, request, context):  # 感知子系统向其他进程提供“获取自车置信图”的服务

        # ###################################################需要真实数据来源
        timestamp = int(time.time())        # 时间戳
        my_conf_map = np.array([4, 5, 6])   # 自车置信图
        # ###################################################

        return Service_pb2.MyConfMap(  # 序列化并返回自车置信图
            timestamp=timestamp,
            my_conf_map=Service_pb2.NdArray(
                data=my_conf_map.tobytes(),
                dtype=str(my_conf_map.dtype),
                shape=list(my_conf_map.shape)
            )
        )

    def GetMyCommMask(self, request, context):  # 感知子系统向其他进程提供“获取自车协作图”的服务

        # ###################################################需要真实数据来源
        timestamp = int(time.time())        # 时间戳
        my_comm_mask = np.array([7, 8, 9])  # 自车协作图
        # ###################################################

        return Service_pb2.MyCommMask(  # 序列化并返回自车协作图
            timestamp=timestamp,
            my_comm_mask=Service_pb2.NdArray(
                data=my_comm_mask.tobytes(),
                dtype=str(my_comm_mask.dtype),
                shape=list(my_comm_mask.shape)
            )
        )

    def GetMyPVAInfo(self, request, context):   # 感知子系统向其他进程提供“获取自车位置、速度、加速度信息”的服务

        # ###################################################需要真实数据来源
        timestamp = int(time.time())            # 时间戳
        pose = np.array([10, 11, 12])           # 位置
        velocity = np.array([13, 14, 15])       # 速度
        acceleration = np.array([16, 17, 18])   # 加速度
        # ###################################################

        return Service_pb2.MyPVAInfo(  # 序列化并返回自车位置、速度、加速度信息
            timestamp=timestamp,
            pose=Service_pb2.NdArray(
                data=pose.tobytes(),
                dtype=str(pose.dtype),
                shape=list(pose.shape)
            ),
            velocity=Service_pb2.NdArray(
                data=velocity.tobytes(),
                dtype=str(velocity.dtype),
                shape=list(velocity.shape)
            ),
            acceleration=Service_pb2.NdArray(
                data=acceleration.tobytes(),
                dtype=str(acceleration.dtype),
                shape=list(acceleration.shape)
            )
        )

    def GetMyExtrinsicMatrix(self, request, context):   # 感知子系统向其他进程提供“获取自车外参矩阵”的服务

        # ###################################################需要真实数据来源
        timestamp = int(time.time())                    # 时间戳
        my_extrinsic_matrix = np.array([19, 20, 21])    # 自车外参矩阵
        # ###################################################

        return Service_pb2.MyExtrinsicMatrix(  # 序列化并返回自车外参矩阵
            timestamp=timestamp,
            my_extrinsic_matrix=Service_pb2.NdArray(
                data=my_extrinsic_matrix.tobytes(),
                dtype=str(my_extrinsic_matrix.dtype),
                shape=list(my_extrinsic_matrix.shape)
            )
        )


class PerceptionServerThread(threading.Thread):    # 感知子系统的Server线程
    def run(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
            ('grpc.max_send_message_length', 64 * 1024 * 1024),                 # 设置gRPC 消息的最大发送和接收大小为64MB
            ('grpc.max_receive_message_length', 64 * 1024 * 1024)])
        Service_pb2_grpc.add_PerceptionServiceServicer_to_server(PerceptionService(), server)
        server.add_insecure_port('[::]:50051')
        server.start()                              # 非阻塞, 会实例化一个新线程来处理请求
        print("Perception Server is up and running on port 50051.")
        try:
            server.wait_for_termination()           # 保持服务器运行直到终止
        except KeyboardInterrupt:
            server.stop(0)                          # 服务器终止
            print("Perception Server terminated.")


class PerceptionClient:                             # 感知子系统的Client类，用于请求其他进程的服务
    pass