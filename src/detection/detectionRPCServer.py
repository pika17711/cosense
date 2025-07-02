import logging
import threading
import grpc
import time
from concurrent import futures

from rpc import Service_pb2_grpc
from rpc import Service_pb2
from utils.sharedInfo import SharedInfo
from utils.rpc_utils import np_to_protobuf, protobuf_to_np, protobuf_to_dict
from utils.detection_utils import pcd_to_spatial_feature, lidar_poses_to_projected_spatial_features, \
    spatial_feature_to_conf_map, spatial_feature_to_pred_box, spatial_feature_to_comm_masked_feature


class DetectionRPCService(Service_pb2_grpc.DetectionServiceServicer):  # 融合检测子系统的Service类
    def __init__(self, shared_info: SharedInfo):
        super().__init__()
        self.shared_info = shared_info

    def GetCommMaskAndLidarPose(self, request, context):
        comm_mask = self.shared_info.get_comm_mask_copy()
        ts_comm_mask = int(time.time())

        lidar_pose = self.shared_info.get_lidar_pose_copy()
        ts_lidar_pose = int(time.time())
        return Service_pb2.CommMaskAndLidarPose(comm_mask=comm_mask, ts_comm_mask=ts_comm_mask,
                                                lidar_pose=lidar_pose, ts_lidar_pose=ts_lidar_pose)

    def GetFusedFeature(self, request, context):  # 融合检测子系统向其他进程提供“获取融合后的特征”的服务
        fused_feature = self.shared_info.get_fused_feature_copy()
        ts_fused_feature = int(time.time())  # 时间戳

        return Service_pb2.Feature(feature=np_to_protobuf(fused_feature),
                                   ts_feature=ts_fused_feature)

    def GetFusedCommMask(self, request, context):  # 融合检测子系统向其他进程提供“获取融合后的协作图”的服务
        fused_comm_mask = self.shared_info.get_fused_comm_mask_copy()
        ts_fused_comm_mask = int(time.time())  # 时间戳

        return Service_pb2.CommMask(comm_mask=np_to_protobuf(fused_comm_mask),
                                    ts_comm_mask=ts_fused_comm_mask)

    def GetLatestPredBox(self, request, context):  # 融合检测子系统向其他进程提供“获取最新检测框”的服务
        pred_box = self.shared_info.get_pred_box_copy()  # 最新检测框
        ts_pred_box = int(time.time())  # 时间戳

        return Service_pb2.PredBox(pred_box=np_to_protobuf(pred_box),
                                   ts_pred_box=ts_pred_box)

    def PCD2Feature(self, request, context):
        pcd = protobuf_to_np(request.pcd)
        ts_pcd = request.ts_pcd
        # 特征
        _, spatial_feature = pcd_to_spatial_feature(pcd, self.shared_info)

        return Service_pb2.Feature(feature=np_to_protobuf(spatial_feature),
                                   ts_feature=ts_pcd)

    def PCD2FeatureAndConfMap(self, request, context):  # 融合检测子系统向其他进程提供“根据点云获取特征和置信图”的服务
        pcd = protobuf_to_np(request.pcd)
        ts_pcd = request.ts_pcd
        # 特征
        _, spatial_feature = pcd_to_spatial_feature(pcd, self.shared_info)
        # 置信图
        conf_map = spatial_feature_to_conf_map(spatial_feature, self.shared_info)

        return Service_pb2.FeatureAndConfMap(feature=np_to_protobuf(spatial_feature),
                                             ts_feature=ts_pcd,
                                             conf_map=np_to_protobuf(conf_map),
                                             ts_conf_map=ts_pcd)

    def LidarPoses2ProjectedFeatures(self, request, context):
        lidar_poses_protobuf = request.lidar_poses

        lidar_poses = protobuf_to_dict(lidar_poses_protobuf)

        my_lidar_pose = self.shared_info.get_lidar_pose_copy()
        my_pcd = self.shared_info.get_pcd_copy()
        projected_features = lidar_poses_to_projected_spatial_features(my_lidar_pose, my_pcd, lidar_poses,
                                                                       self.shared_info)

        projected_features_protobuf = {}
        for cav_id, projected_feature in projected_features.items():
            projected_feature_protobuf = Service_pb2.Feature(feature=np_to_protobuf(projected_feature['feature']),
                                                             ts_feature=projected_feature['ts_feature'])
            projected_features_protobuf[cav_id] = projected_feature_protobuf

        return Service_pb2.Features(features=projected_features_protobuf)

    def LidarPoses2ProjectedCommMaskedFeatures(self, request, context):
        lidar_poses_protobuf = request.lidar_poses

        lidar_poses = protobuf_to_dict(lidar_poses_protobuf)

        my_lidar_pose = self.shared_info.get_lidar_pose_copy()
        my_pcd = self.shared_info.get_pcd_copy()
        projected_features = lidar_poses_to_projected_spatial_features(my_lidar_pose, my_pcd, lidar_poses,
                                                                       self.shared_info)

        projected_comm_masked_features_protobuf = {}
        for cav_id, projected_feature in projected_features.items():
            comm_masked_feature, comm_mask = spatial_feature_to_comm_masked_feature(projected_feature['feature'], self.shared_info)

            projected_comm_masked_feature_protobuf = Service_pb2.CommMaskedFeature(comm_masked_feature=np_to_protobuf(comm_masked_feature),
                                                                                   comm_mask=np_to_protobuf(comm_mask),
                                                                                   ts_feature=projected_feature['ts_feature'])
            projected_comm_masked_features_protobuf[cav_id] = projected_comm_masked_feature_protobuf

        return Service_pb2.CommMaskedFeatures(comm_masked_features=projected_comm_masked_features_protobuf)

    def Feature2ConfMap(self, request, context):  # 融合检测子系统向其他进程提供“根据特征获取置信图”的服务
        spatial_feature = protobuf_to_np(request.feature)
        ts_spatial_feature = request.ts_feature
        # 置信图
        conf_map = spatial_feature_to_conf_map(spatial_feature, self.shared_info)

        return Service_pb2.ConfMap(conf_map=np_to_protobuf(conf_map),
                                   ts_conf_map=ts_spatial_feature)

    def Feature2PredBox(self, request, context):  # 融合检测子系统向其他进程提供“根据特征获取检测框”的服务
        spatial_feature = protobuf_to_np(request.feature)
        ts_spatial_feature = request.feature
        # 检测框
        pred_box = spatial_feature_to_pred_box(spatial_feature, self.shared_info)

        return Service_pb2.PredBox(pred_box=np_to_protobuf(pred_box),
                                   ts_pred_box=ts_spatial_feature)


class DetectionServerThread:  # 融合检测子系统的Server线程
    def __init__(self, shared_info):
        self.shared_info = shared_info
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
            ('grpc.max_send_message_length', 64 * 1024 * 1024),  # 设置gRPC 消息的最大发送和接收大小为64MB
            ('grpc.max_receive_message_length', 64 * 1024 * 1024)])
        Service_pb2_grpc.add_DetectionServiceServicer_to_server(DetectionRPCService(self.shared_info), self.server)
        self.stop_event = threading.Event()
        self.run_thread = threading.Thread(target=self.run, name='detection rpc server', daemon=True)

    def run(self):
        self.server.add_insecure_port('[::]:50053')
        self.server.start()  # 非阻塞, 会实例化一个新线程来处理请求
        logging.info("Detection Server is up and running on port 50053.")
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
