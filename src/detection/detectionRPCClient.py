import logging
import grpc

from rpc import Service_pb2
from rpc import Service_pb2_grpc
from utils.rpc_utils import protobuf_to_np, np_to_protobuf, protobuf_to_dict


class DetectionRPCClient:  # 融合检测子系统的Client类，用于向融合检测子系统的服务器请求服务
    def __init__(self):
        detection_channel = grpc.insecure_channel('localhost:50053', options=[  # 与融合检测子系统的服务器建立连接
            ('grpc.max_send_message_length', 64 * 1024 * 1024),  # 设置gRPC 消息的最大发送和接收大小为64MB
            ('grpc.max_receive_message_length', 64 * 1024 * 1024)])
        self.__detection_stub = Service_pb2_grpc.DetectionServiceStub(detection_channel)

    def get_comm_mask_and_lidar_pose(self):
        try:
            response = self.__detection_stub.GetCommMaskAndLidarPose(Service_pb2.Empty(), timeout=5)  # 请求融合检测子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC get_comm_mask_and_lidar_pose failed: code={e.code()}")  # 记录grpc异常
            return None, None, None, None

        # 协作图
        comm_mask = protobuf_to_np(response.comm_mask)
        ts_comm_mask = response.ts_comm_mask  # 时间戳
        # 雷达位姿
        lidar_pose = protobuf_to_np(response.lidar_pose)
        ts_lidar_pose = response.ts_lidar_pose

        return comm_mask, ts_comm_mask, lidar_pose, ts_lidar_pose

    def get_fused_spatial_feature(self):  # 从融合检测子系统获取融合后的特征
        try:
            response = self.__detection_stub.GetFusedFeature(Service_pb2.Empty(), timeout=5)  # 请求融合检测子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC get_fused_spatial_feature failed: code={e.code()}")  # 记录grpc异常
            return None, None

        # 融合后的特征
        fused_spatial_feature = protobuf_to_np(response.feature)
        ts_fused_spatial_feature = response.ts_feature  # 时间戳
        return fused_spatial_feature, ts_fused_spatial_feature

    def get_fused_comm_mask(self):  # 从融合检测子系统获取融合后的协作图
        try:
            response = self.__detection_stub.GetFusedCommMask(Service_pb2.Empty(), timeout=5)  # 请求融合检测子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC get_fused_comm_mask failed: code={e.code()}")  # 记录grpc异常
            return None, None

        # 融合后的协作图
        fused_comm_mask = protobuf_to_np(response.comm_mask)
        ts_fused_comm_mask = response.ts_comm_mask  # 时间戳
        return fused_comm_mask, ts_fused_comm_mask

    def get_latest_pred_box(self):  # 从融合检测子系统获取最新检测框
        try:
            response = self.__detection_stub.GetLatestPredBox(Service_pb2.Empty(), timeout=5)  # 请求融合检测子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC get_latest_pred_box failed: code={e.code()}")  # 记录grpc异常
            return None, None

        # 最新检测框
        pred_box = protobuf_to_np(response.pred_box)
        ts_pred_box = response.ts_pred_box  # 时间戳
        return pred_box, ts_pred_box

    def pcd_to_spatial_feature(self, ts_pcd, pcd):  # 融合检测子系统根据点云返回特征
        request = Service_pb2.PCD(pcd=np_to_protobuf(pcd),
                                  ts_pcd=ts_pcd)
        try:
            response = self.__detection_stub.PCD2Feature(request, timeout=10)  # 请求融合检测子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC pcd_to_spatial_feature failed: code={e.code()}")  # 记录grpc异常
            return None, None

        # 特征
        spatial_feature = protobuf_to_np(response.feature)
        ts_spatial_feature = response.ts_feature  # 时间戳
        return spatial_feature, ts_spatial_feature

    def pcd_to_spatial_feature_and_conf_map(self, ts_pcd, pcd):  # 融合检测子系统根据点云返回特征和置信图
        request = Service_pb2.PCD(pcd=np_to_protobuf(pcd),
                                  ts_pcd=ts_pcd)
        try:
            response = self.__detection_stub.PCD2FeatureAndConfMap(request, timeout=10)  # 请求融合检测子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC pcd_to_spatial_feature_and_conf_map failed: code={e.code()}")  # 记录grpc异常
            return None, None, None, None

        # 特征
        spatial_feature = protobuf_to_np(response.feature)
        ts_spatial_feature = response.ts_feature
        # 置信图
        conf_map = protobuf_to_np(response.conf_map)
        ts_conf_map = response.ts_conf_map
        return spatial_feature, ts_spatial_feature, conf_map, ts_conf_map

    def lidar_pose_to_projected_spatial_feature(self, lidar_pose):
        pass

    def lidar_poses_to_projected_spatial_features(self, lidar_poses):
        lidar_poses_protobuf = {}
        for cav_id, lidar_pose in lidar_poses.items():
            lidar_pose_protobuf = Service_pb2.LidarPoses.LidarPose(lidar_pose=np_to_protobuf(lidar_pose['lidar_pose']),
                                                                   ts_lidar_pose=lidar_pose['ts_lidar_pose'])
            lidar_poses_protobuf[cav_id] = lidar_pose_protobuf

        request = Service_pb2.LidarPoses(lidar_poses=lidar_poses_protobuf)
        try:
            response = self.__detection_stub.LidarPoses2ProjectedFeatures(request, timeout=10)  # 请求融合检测子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC lidar_poses_to_projected_spatial_features failed: code={e.code()}")  # 记录grpc异常
            return None

        projected_spatial_features_protobuf = response.features
        projected_spatial_features = protobuf_to_dict(projected_spatial_features_protobuf)

        return projected_spatial_features

    def lidar_poses_to_projected_comm_masked_features(self, lidar_poses):
        lidar_poses_protobuf = {}
        for cav_id, lidar_pose in lidar_poses.items():
            lidar_pose_protobuf = Service_pb2.LidarPoses.LidarPose(lidar_pose=np_to_protobuf(lidar_pose['lidar_pose']),
                                                                   ts_lidar_pose=lidar_pose['ts_lidar_pose'])
            lidar_poses_protobuf[cav_id] = lidar_pose_protobuf

        request = Service_pb2.LidarPoses(lidar_poses=lidar_poses_protobuf)
        try:
            response = self.__detection_stub.LidarPoses2ProjectedCommMaskedFeatures(request, timeout=10)  # 请求融合检测子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC lidar_poses_to_projected_comm_masked_features failed: code={e.code()}")  # 记录grpc异常
            return None

        projected_comm_masked_features_protobuf = response.comm_masked_featrues
        projected_comm_masked_features = protobuf_to_dict(projected_comm_masked_features_protobuf)

        return projected_comm_masked_features

    def request_map_to_projected_comm_masked_feature(self, lidar_pose, request_map, ts_request_map):
        request = Service_pb2.RequestMap(lidar_pose=np_to_protobuf(lidar_pose),
                                         request_map=np_to_protobuf(request_map),
                                         ts_request_map=ts_request_map)

        try:
            response = self.__detection_stub.RequestMap2ProjectedCommMaskedFeature(request, timeout=10)  # 请求融合检测子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC request_map_to_projected_comm_masked_feature failed: code={e.code()}")  # 记录grpc异常
            return None, None, None

        comm_masked_feature = protobuf_to_np(response.comm_masked_feature)
        comm_mask = protobuf_to_np(response.comm_mask)
        ts_feature = response.ts_feature

        return comm_masked_feature, comm_mask, ts_feature

    def spatial_feature_to_conf_map(self, ts_spatial_feature, spatial_feature):  # 融合检测子系统根据特征返回置信图
        request = Service_pb2.Feature(feature=np_to_protobuf(spatial_feature),
                                      ts_feature=ts_spatial_feature)
        try:
            response = self.__detection_stub.Feature2ConfMap(request, timeout=10)  # 请求融合检测子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC spatial_feature_to_conf_map failed: code={e.code()}")  # 记录grpc异常
            return None, None

        # 置信图
        conf_map = protobuf_to_np(response.conf_map)
        ts_conf_map = response.ts_conf_map  # 时间戳
        return conf_map, ts_conf_map

    def spatial_feature_to_pred_box(self, ts_spatial_feature, spatial_feature):  # 融合检测子系统根据特征返回检测框
        request = Service_pb2.Feature(feature=np_to_protobuf(spatial_feature),
                                      ts_feature=ts_spatial_feature)
        try:
            response = self.__detection_stub.Feature2PredBox(request, timeout=10)  # 请求融合检测子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC spatial_feature_to_pred_box failed: code={e.code()}")  # 记录grpc异常
            return None, None

        # 检测框
        pred_box = protobuf_to_np(response.pred_box)
        ts_pred_bpx = response.ts_pred_box  # 时间戳
        return pred_box, ts_pred_bpx
