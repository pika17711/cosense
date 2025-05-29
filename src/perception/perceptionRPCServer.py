import threading
import logging

import grpc
from concurrent import futures
from rpc import Service_pb2
from rpc import Service_pb2_grpc
import time
import numpy as np
from utils.sharedInfo import SharedInfo


class PerceptionRPCService(Service_pb2_grpc.PerceptionServiceServicer):  # 感知子系统的Service类
    def __init__(self, my_info: SharedInfo):
        self.my_info = my_info

    def GetMyPCD(self, request, context):           # 感知子系统向其他进程提供“获取自车点云”的服务
        timestamp = int(time.time())        # 时间戳
        my_pcd = self.my_info.get_pcd_copy()

        return Service_pb2.PCD(  # 序列化并返回自车点云
            timestamp=timestamp,
            pcd=Service_pb2.NdArray(
                data=my_pcd.tobytes(),
                dtype=str(my_pcd.dtype),
                shape=list(my_pcd.shape)
            )
        )

    def GetMyPoseAndPCD(self, request, context):        # 感知子系统向其他进程提供“获取自车雷达位姿和点云”的服务
        timestamp = int(time.time())  # 时间戳
        my_pose = self.my_info.get_pose_copy()
        my_pcd = self.my_info.get_pcd_copy()

        return Service_pb2.PoseAndPCD(  # 序列化并返回自车点云
            timestamp=timestamp,
            pose=Service_pb2.NdArray(
                data=my_pose.tobytes(),
                dtype=str(my_pose.dtype),
                shape=list(my_pose.shape)
            ),
            pcd=Service_pb2.NdArray(
                data=my_pcd.tobytes(),
                dtype=str(my_pcd.dtype),
                shape=list(my_pcd.shape)
            )
        )

    def GetMyFeature(self, request, context):  # 感知子系统向其他进程提供“获取自车特征”的服务
        timestamp = int(time.time())                        # 时间戳
        my_feature = self.my_info.get_feature_copy()

        return Service_pb2.Feature(  # 序列化并返回自车特征
            timestamp=timestamp,
            feature=Service_pb2._Feature(
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
        timestamp = int(time.time())        # 时间戳
        my_conf_map = self.my_info.get_conf_map_copy()

        return Service_pb2.ConfMap(  # 序列化并返回自车置信图
            timestamp=timestamp,
            conf_map=Service_pb2.NdArray(
                data=my_conf_map.tobytes(),
                dtype=str(my_conf_map.dtype),
                shape=list(my_conf_map.shape)
            )
        )

    def GetMyCommMask(self, request, context):  # 感知子系统向其他进程提供“获取自车协作图”的服务
        timestamp = int(time.time())        # 时间戳
        my_comm_mask = self.my_info.get_comm_mask_copy()

        return Service_pb2.CommMask(  # 序列化并返回自车协作图
            timestamp=timestamp,
            comm_mask=Service_pb2.NdArray(
                data=my_comm_mask.tobytes(),
                dtype=str(my_comm_mask.dtype),
                shape=list(my_comm_mask.shape)
            )
        )

    def GetMyPVAInfo(self, request, context):   # 感知子系统向其他进程提供“获取自车位置、速度、加速度信息”的服务
        timestamp = int(time.time())            # 时间戳
        pose = self.my_info.get_pose_copy()
        velocity = self.my_info.get_velocity_copy()
        acceleration = self.my_info.get_acceleration_copy()

        return Service_pb2.PVAInfo(  # 序列化并返回自车位置、速度、加速度信息
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
        timestamp = int(time.time())                    # 时间戳
        my_extrinsic_matrix = self.my_info.get_extrinsic_matrix_copy()

        return Service_pb2.ExtrinsicMatrix(  # 序列化并返回自车外参矩阵
            timestamp=timestamp,
            extrinsic_matrix=Service_pb2.NdArray(
                data=my_extrinsic_matrix.tobytes(),
                dtype=str(my_extrinsic_matrix.dtype),
                shape=list(my_extrinsic_matrix.shape)
            )
        )


class PerceptionServerThread:                                 # 感知子系统的Server线程
    def __init__(self, my_info):
        self.my_info = my_info
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
            ('grpc.max_send_message_length', 64 * 1024 * 1024),  # 设置gRPC 消息的最大发送和接收大小为64MB
            ('grpc.max_receive_message_length', 64 * 1024 * 1024)])
        Service_pb2_grpc.add_PerceptionServiceServicer_to_server(PerceptionRPCService(self.my_info), self.server)
        self.stop_event = threading.Event()
        self.run_thread = threading.Thread(target=self.run, name='perception rpc server', daemon=True)

    def run(self):
        self.server.add_insecure_port('[::]:50051')
        self.server.start()                              # 非阻塞, 会实例化一个新线程来处理请求
        logging.info("Perception Server is up and running on port 50051.")
        try:
            # 等待停止事件或被中断
            while not self.stop_event.is_set():
                self.stop_event.wait(1)  # 每1秒检查一次停止标志
        except KeyboardInterrupt:
            pass
        finally:
            # 优雅地关闭服务器
            if self.server:
                self.server.stop(0.5).wait()

    def start(self):
        self.run_thread.start()

    def close(self):
        self.stop_event.set()  # 设置停止标志
