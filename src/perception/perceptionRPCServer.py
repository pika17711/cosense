import threading
import logging

import grpc
from concurrent import futures
import numpy as np
from rpc import Service_pb2
from rpc import Service_pb2_grpc
from utils.rpc_utils import np_to_protobuf
import time
from utils.sharedInfo import SharedInfo


class PerceptionRPCService(Service_pb2_grpc.PerceptionServiceServicer):  # 感知子系统的Service类
    def __init__(self, my_info: SharedInfo):
        self.my_info = my_info

    def GetMyPCD(self, request, context):  # 感知子系统向其他进程提供“获取自车点云”的服务
        my_pcd = self.my_info.get_pcd_copy()
        ts_pcd = int(time.time())  # 时间戳

        return Service_pb2.PCD(pcd=np_to_protobuf(my_pcd),
                               ts_pcd=ts_pcd)

    def GetMyLidarPoseAndPCD(self, request, context):  # 感知子系统向其他进程提供“获取自车雷达位姿和点云”的服务
        my_lidar_pose = self.my_info.get_lidar_pose_copy()
        ts_lidar_pose = int(time.time())
        my_pcd = self.my_info.get_pcd_copy()
        ts_pcd = ts_lidar_pose

        return Service_pb2.LidarPoseAndPCD(lidar_pose=np_to_protobuf(my_lidar_pose),
                                           ts_lidar_pose=ts_lidar_pose,
                                           pcd=np_to_protobuf(my_pcd),
                                           ts_pcd=ts_pcd)

    def GetMyPVA(self, request, context):  # 感知子系统向其他进程提供“获取自车位置、速度、加速度信息”的服务
        lidar_pose = self.my_info.get_lidar_pose_copy()
        ts_lidar_pose = int(time.time())
        # velocity = self.my_info.get_velocity_copy()
        velocity = self.my_info.get_speed_copy()
        velocity = np.array(velocity if velocity is not None else -1)
        ts_v = ts_lidar_pose
        acceleration = self.my_info.get_acceleration_copy()
        acceleration = np.array(acceleration if acceleration is not None else -1)
        ts_a = ts_lidar_pose

        return Service_pb2.PVA(lidar_pose=np_to_protobuf(lidar_pose),
                               ts_lidar_pose=ts_lidar_pose,
                               velocity=np_to_protobuf(velocity),
                               ts_v=ts_v,
                               acceleration=np_to_protobuf(acceleration),
                               ts_a=ts_a)

    def GetMyExtrinsicMatrix(self, request, context):  # 感知子系统向其他进程提供“获取自车外参矩阵”的服务
        my_extrinsic_matrix = self.my_info.get_extrinsic_matrix_copy()
        ts_extrinsic_matrix = int(time.time())  # 时间戳

        return Service_pb2.ExtrinsicMatrix(extrinsic_matrix=np_to_protobuf(my_extrinsic_matrix),
                                           ts_extrinsic_matrix=ts_extrinsic_matrix)


class PerceptionServerThread:  # 感知子系统的Server线程
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
        self.server.start()  # 非阻塞, 会实例化一个新线程来处理请求
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
