import threading
import grpc
from concurrent import futures
import logging

from appConfig import AppConfig
from rpc import Service_pb2
from rpc import Service_pb2_grpc
from utils.rpc_utils import np_to_protobuf
from utils.othersInfos import OthersInfos


class CollaborationRPCService(Service_pb2_grpc.CollaborationServiceServicer):  # 协同感知子系统的Service类
    def __init__(self, cfg: AppConfig, others_infos: OthersInfos):
        self.cfg = cfg
        self.others_infos = others_infos

    def GetOthersInfos(self, request, context):  # 协同感知子系统向其他进程提供“获取所有他车信息”的服务
        others_infos = self.others_infos.get_infos()
        # others_infos = self.others_infos.pop_infos()

        others_infos_protobuf = {}

        for cav_id, cav_info in others_infos.items():
            cav_info_protobuf = Service_pb2.OthersInfos.CAVInfo(
                lidar_pose=np_to_protobuf(cav_info['lidar_pose']),
                ts_lidar_pose=cav_info['ts_lidar_pose'],
                velocity=np_to_protobuf(cav_info['velocity']),
                ts_v=cav_info['ts_v'],
                acceleration=np_to_protobuf(cav_info['acceleration']),
                ts_a=cav_info['ts_a'],
                feature=np_to_protobuf(cav_info['feature']),
                ts_feature=cav_info['ts_feature'])

            if 'comm_mask' in cav_info and cav_info['comm_mask'] is not None:
                cav_info_protobuf.comm_mask.CopyFrom(np_to_protobuf(cav_info['comm_mask']))

            others_infos_protobuf[cav_id] = cav_info_protobuf

        return Service_pb2.OthersInfos(others_infos=others_infos_protobuf)

    def GetOthersCommMasks(self, request, context):  # 协同感知子系统向其他进程提供“获取所有他车协作图”的服务
        # others_infos = self.others_infos.get_infos()
        others_infos = self.others_infos.pop_infos()

        others_comm_masks = {}

        for cav_id, cav_info in others_infos.items():
            cav_comm_mask = Service_pb2.CommMask(comm_mask=np_to_protobuf(cav_info['comm_mask']),
                                                 ts_comm_mask=cav_info['ts_comm_mask'])

            others_comm_masks[cav_id] = cav_comm_mask

        return Service_pb2.OthersCommMasks(others_comm_masks=others_comm_masks)

    def GetOthersLidarPosesAndPCDs(self, request, context):
        others_infos = self.others_infos.get_infos()
        # others_infos = self.others_infos.pop_infos()

        others_lidar_poses_and_pcds = {}

        for cav_id, cav_info in others_infos.items():
            cav_lidar_pose_and_pcd = Service_pb2.LidarPoseAndPCD(lidar_pose=np_to_protobuf(cav_info['lidar_pose']),
                                                                 ts_lidar_pose=cav_info['ts_lidar_pose'],
                                                                 pcd=np_to_protobuf(cav_info['pcd']),
                                                                 ts_pcd=cav_info['ts_pcd'])

            others_lidar_poses_and_pcds[cav_id] = cav_lidar_pose_and_pcd

        return Service_pb2.LidarPosesAndPCDs(others_lidar_poses_and_pcds=others_lidar_poses_and_pcds)


class CollaborationRPCServerThread:  # 协同感知子系统的Server线程
    def __init__(self, cfg: AppConfig, others_infos):
        self.cfg = cfg
        self.others_infos = others_infos
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
            ('grpc.max_send_message_length', 64 * 1024 * 1024),  # 设置gRPC 消息的最大发送和接收大小为64MB
            ('grpc.max_receive_message_length', 64 * 1024 * 1024)])
        Service_pb2_grpc.add_CollaborationServiceServicer_to_server(CollaborationRPCService(self.cfg, self.others_infos),
                                                                    self.server)
        self.stop_event = threading.Event()
        self.run_thread = threading.Thread(target=self.run, name='collaboration rpc server', daemon=True)

    def run(self):
        self.server.add_insecure_port('[::]:50052')
        self.server.start()  # 非阻塞, 会实例化一个新线程来处理请求
        logging.info("Collaboration Server is up and running on port 50052.")
        try:
            # 等待停止事件或被中断
            while not self.stop_event.is_set():
                self.stop_event.wait(1)  # 每1秒检查一次停止标志
        except KeyboardInterrupt:
            pass
        finally:
            # 优雅地关闭服务器
            if self.server:
                self.server.stop(self.cfg.close_timeout).wait()

    def start(self):
        self.run_thread.start()

    def close(self):
        self.stop_event.set()  # 设置停止标志
