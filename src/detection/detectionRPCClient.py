import logging
import grpc
import numpy as np

from rpc import Service_pb2
from rpc import Service_pb2_grpc


class DetectionRPCClient:                      # 融合检测子系统的Client类，用于向融合检测子系统的服务器请求服务
    def __init__(self):
        detection_channel = grpc.insecure_channel('localhost:50053', options=[      # 与融合检测子系统的服务器建立连接
            ('grpc.max_send_message_length', 64 * 1024 * 1024),                     # 设置gRPC 消息的最大发送和接收大小为64MB
            ('grpc.max_receive_message_length', 64 * 1024 * 1024)])
        self.__detection_stub = Service_pb2_grpc.DetectionServiceStub(detection_channel)

    def get_fused_feature(self):  # 从融合检测子系统获取融合后的特征
        try:
            response = self.__detection_stub.GetFusedFeature(Service_pb2.Empty(), timeout=5)  # 请求融合检测子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code}, details={e.details}")  # 记录grpc异常
            return -1, -1

        timestamp = response.timestamp  # 时间戳
        # 体素特征
        voxel_features_message = response.feature.voxel_features
        voxel_features = np.frombuffer(voxel_features_message.data,
                                       dtype=voxel_features_message.dtype).reshape(voxel_features_message.shape)
        # 体素坐标
        voxel_coords_message = response.feature.voxel_coords
        voxel_coords = np.frombuffer(voxel_coords_message.data,
                                     dtype=voxel_coords_message.dtype).reshape(voxel_coords_message.shape)

        # 体素点数
        voxel_num_points_message = response.feature.voxel_num_points
        voxel_num_points = np.frombuffer(voxel_num_points_message.data,
                                         dtype=voxel_num_points_message.dtype).reshape(voxel_num_points_message.shape)
        # 融合后的特征
        fused_feature = {'voxel_features': voxel_features,
                         'voxel_coords': voxel_coords,
                         'voxel_num_points': voxel_num_points}
        return timestamp, fused_feature

    def get_fused_comm_mask(self):  # 从融合检测子系统获取融合后的协作图
        try:
            response = self.__detection_stub.GetFusedCommMask(Service_pb2.Empty(), timeout=5)  # 请求融合检测子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code}, details={e.details}")  # 记录grpc异常
            return -1, -1

        timestamp = response.timestamp  # 时间戳
        # 融合后的协作图
        fused_comm_mask = np.frombuffer(response.comm_mask.data,
                                        dtype=response.comm_mask.dtype).reshape(response.comm_mask.shape)
        return timestamp, fused_comm_mask

    def get_latest_pred_box(self):  # 从融合检测子系统获取最新检测框
        try:
            response = self.__detection_stub.GetLatestPredBox(Service_pb2.Empty(), timeout=5)  # 请求融合检测子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code}, details={e.details}")  # 记录grpc异常
            return -1, -1

        timestamp = response.timestamp  # 时间戳
        # 最新检测框
        pred_box = np.frombuffer(response.pred_box.data,
                                 dtype=response.pred_box.dtype).reshape(response.pred_box.shape)
        return timestamp, pred_box

    def pcd2feature(self, timestamp, pcd):                  # 融合检测子系统根据点云返回特征
        request = Service_pb2.PCD(  # 序列化点云请求
            timestamp=timestamp,
            pcd=Service_pb2.NdArray(
                data=pcd.tobytes(),
                dtype=str(pcd.dtype),
                shape=list(pcd.shape)
            )
        )
        try:
            response = self.__detection_stub.PCD2Feature(request, timeout=10)  # 请求融合检测子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code}, details={e.details}")  # 记录grpc异常
            return -1, -1

        timestamp = response.timestamp  # 时间戳

        # 体素特征
        voxel_features_message = response.feature.voxel_features
        voxel_features = np.frombuffer(voxel_features_message.data,
                                       dtype=voxel_features_message.dtype).reshape(voxel_features_message.shape)
        # 体素坐标
        voxel_coords_message = response.feature.voxel_coords
        voxel_coords = np.frombuffer(voxel_coords_message.data,
                                     dtype=voxel_coords_message.dtype).reshape(voxel_coords_message.shape)

        # 体素点数
        voxel_num_points_message = response.feature.voxel_num_points
        voxel_num_points = np.frombuffer(voxel_num_points_message.data,
                                         dtype=voxel_num_points_message.dtype).reshape(voxel_num_points_message.shape)
        # 特征
        feature = {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
        }
        return timestamp, feature

    def poses2projected_features(self, timestamps, poses):
        request = Service_pb2.Poses(  # 序列化雷达位姿
            timestamps=timestamps,
            poses=Service_pb2.NdArray(
                data=poses.tobytes(),
                dtype=str(poses.dtype),
                shape=list(poses.shape)
            )
        )
        try:
            response = self.__detection_stub.Poses2ProjectedFeatures(request, timeout=10)  # 请求融合检测子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code}, details={e.details}")  # 记录grpc异常
            return -1, -1

        timestamps = response.timestamps  # 时间戳
        features_lens = response.features_lens
        # 体素特征
        voxel_features_message = response.feature.voxel_features
        voxel_features = np.frombuffer(voxel_features_message.data,
                                       dtype=voxel_features_message.dtype).reshape(voxel_features_message.shape)
        # 体素坐标
        voxel_coords_message = response.feature.voxel_coords
        voxel_coords = np.frombuffer(voxel_coords_message.data,
                                     dtype=voxel_coords_message.dtype).reshape(voxel_coords_message.shape)

        # 体素点数
        voxel_num_points_message = response.feature.voxel_num_points
        voxel_num_points = np.frombuffer(voxel_num_points_message.data,
                                         dtype=voxel_num_points_message.dtype).reshape(voxel_num_points_message.shape)
        features = []

        start_index = 0

        for feature_len in features_lens:
            end_index = start_index + feature_len

            feature = {'voxel_features': voxel_features[start_index: end_index],
                       'voxel_coords': voxel_coords[start_index: end_index],
                       'voxel_num_points': voxel_num_points[start_index: end_index]}
            features.append(feature)

            start_index = end_index
        return timestamps, features

    def pcd2feature_and_conf_map(self, timestamp, pcd):  # 融合检测子系统根据点云返回特征和置信图
        request = Service_pb2.PCD(  # 序列化点云请求
            timestamp=timestamp,
            pcd=Service_pb2.NdArray(
                data=pcd.tobytes(),
                dtype=str(pcd.dtype),
                shape=list(pcd.shape)
            )
        )
        try:
            response = self.__detection_stub.PCD2FeatureAndConfMap(request, timeout=10)  # 请求融合检测子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code}, details={e.details}")  # 记录grpc异常
            return -1, -1, -1

        timestamp = response.timestamp  # 时间戳

        # 体素特征
        voxel_features_message = response.feature.voxel_features
        voxel_features = np.frombuffer(voxel_features_message.data,
                                       dtype=voxel_features_message.dtype).reshape(voxel_features_message.shape)
        # 体素坐标
        voxel_coords_message = response.feature.voxel_coords
        voxel_coords = np.frombuffer(voxel_coords_message.data,
                                     dtype=voxel_coords_message.dtype).reshape(voxel_coords_message.shape)

        # 体素点数
        voxel_num_points_message = response.feature.voxel_num_points
        voxel_num_points = np.frombuffer(voxel_num_points_message.data,
                                         dtype=voxel_num_points_message.dtype).reshape(voxel_num_points_message.shape)
        # 特征
        feature = {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
        }
        # 置信图
        conf_map = np.frombuffer(response.conf_map.data,
                                 dtype=response.conf_map.dtype).reshape(response.conf_map.shape)
        return timestamp, feature, conf_map

    def feature2conf_map(self, timestamp, feature):         # 融合检测子系统根据特征返回置信图
        request = Service_pb2.Feature(  # 序列化特征请求
            timestamp=timestamp,
            feature=Service_pb2._Feature(
                voxel_features=Service_pb2.NdArray(
                    data=feature['voxel_features'].tobytes(),
                    dtype=str(feature['voxel_features'].dtype),
                    shape=list(feature['voxel_features'].shape)
                ),
                voxel_coords=Service_pb2.NdArray(
                    data=feature['voxel_coords'].tobytes(),
                    dtype=str(feature['voxel_coords'].dtype),
                    shape=list(feature['voxel_coords'].shape)
                ),
                voxel_num_points=Service_pb2.NdArray(
                    data=feature['voxel_num_points'].tobytes(),
                    dtype=str(feature['voxel_num_points'].dtype),
                    shape=list(feature['voxel_num_points'].shape)
                )
            )
        )
        try:
            response = self.__detection_stub.Feature2ConfMap(request, timeout=10)  # 请求融合检测子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code}, details={e.details}")  # 记录grpc异常
            return -1, -1

        timestamp = response.timestamp  # 时间戳
        # 置信图
        conf_map = np.frombuffer(response.conf_map.data,
                                 dtype=response.conf_map.dtype).reshape(response.conf_map.shape)
        return timestamp, conf_map

    def feature2pred_box(self, timestamp, feature):     # 融合检测子系统根据特征返回检测框
        request = Service_pb2.Feature(  # 序列化特征请求
            timestamp=timestamp,
            feature=Service_pb2._Feature(
                voxel_features=Service_pb2.NdArray(
                    data=feature['voxel_features'].tobytes(),
                    dtype=str(feature['voxel_features'].dtype),
                    shape=list(feature['voxel_features'].shape)
                ),
                voxel_coords=Service_pb2.NdArray(
                    data=feature['voxel_coords'].tobytes(),
                    dtype=str(feature['voxel_coords'].dtype),
                    shape=list(feature['voxel_coords'].shape)
                ),
                voxel_num_points=Service_pb2.NdArray(
                    data=feature['voxel_num_points'].tobytes(),
                    dtype=str(feature['voxel_num_points'].dtype),
                    shape=list(feature['voxel_num_points'].shape)
                )
            )
        )
        try:
            response = self.__detection_stub.Feature2PredBox(request, timeout=10)  # 请求融合检测子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code}, details={e.details}")  # 记录grpc异常
            return -1, -1

        timestamp = response.timestamp  # 时间戳
        #检测框
        pred_box = np.frombuffer(response.pred_box.data,
                                 dtype=response.pred_box.dtype).reshape(response.pred_box.shape)
        return timestamp, pred_box
