import logging
import threading

import grpc
from concurrent import futures

import numpy as np

from rpc import Service_pb2
from rpc import Service_pb2_grpc
import time
from opencood.utils.pcd_utils import mask_points_by_range, mask_ego_points, shuffle_points
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.tools import train_utils
import torch


def pcd2feature(pcd, hypes):        # 根据pcd点云数据获取特征
    pcd = shuffle_points(pcd)
    pcd = mask_ego_points(pcd)
    pcd = mask_points_by_range(pcd, hypes['preprocess']['cav_lidar_range'])

    pre_processor = build_preprocessor(hypes['preprocess'], False)
    feature = pre_processor.preprocess(pcd)
    return feature


def feature2conf_map(feature, model, device):       # 根据特征获取置信图
    voxel_features = torch.from_numpy(feature['voxel_features'])

    voxel_coords = np.pad(feature['voxel_coords'], ((0, 0), (1, 0)), mode='constant', constant_values=0)
    voxel_coords = torch.from_numpy(voxel_coords)

    voxel_num_points = torch.from_numpy(feature['voxel_num_points'])

    record_len = torch.empty(0, dtype=torch.int32)

    pairwise_t_matrix = torch.zeros((1, 5, 5, 4, 4), dtype=torch.float64)

    batch_dict = {'voxel_features': voxel_features,
                  'voxel_coords': voxel_coords,
                  'voxel_num_points': voxel_num_points,
                  'record_len': record_len}

    with torch.no_grad():
        batch_dict = train_utils.to_device(batch_dict, device)

    # n, 4 -> n, c
    batch_dict = model.pillar_vfe(batch_dict)
    # n, c -> N, C, H, W
    batch_dict = model.scatter(batch_dict)
    batch_dict = model.backbone(batch_dict)

    # N, C, H', W': [N, 256, 48, 176] [1:2]
    spatial_features_2d = batch_dict['spatial_features_2d']
    # Down-sample feature to reduce memory
    if model.shrink_flag:
        spatial_features_2d = model.shrink_conv(spatial_features_2d)

    psm_single = model.cls_head(spatial_features_2d)

    # Compressor
    if model.compression:
        # The ego feature is also compressed
        spatial_features_2d = model.naive_compressor(spatial_features_2d)

    if model.multi_scale:
        # Bypass communication cost, communicate at high resolution, neither shrink nor compress
        _, _, conf_map = model.fusion_net(batch_dict['spatial_features'],
                                          psm_single,
                                          record_len,
                                          pairwise_t_matrix,
                                          model.backbone)
    else:
        _, _, conf_map = model.fusion_net(spatial_features_2d,
                                          psm_single,
                                          record_len,
                                          pairwise_t_matrix)
    return conf_map.cpu().data.numpy()


class DetectionService(Service_pb2_grpc.DetectionServiceServicer):  # 融合检测子系统的Service类
    def __init__(self, model, device, hypes, pred_box):
        super().__init__()
        self.model = model
        self.device = device
        self.hypes = hypes
        self.pred_box = pred_box

    def GetFusedFeature(self, request, context):  # 融合检测子系统向其他进程提供“获取融合后的特征”的服务
        # ###################################################需要真实数据来源
        timestamp = int(time.time())                        # 时间戳
        fused_feature = {                                   # 融合后的特征
            'voxel_features': np.array([301, 302, 303]),
            'voxel_coords': np.array([304, 305, 306]),
            'voxel_num_points': np.array([307, 308, 309])
        }
        # ###################################################

        return Service_pb2.FusedFeature(                       # 序列化并返回融合后的特征
            timestamp=timestamp,
            fused_feature=Service_pb2.Feature(
                voxel_features=Service_pb2.NdArray(
                    data=fused_feature['voxel_features'].tobytes(),
                    dtype=str(fused_feature['voxel_features'].dtype),
                    shape=list(fused_feature['voxel_features'].shape)
                ),
                voxel_coords=Service_pb2.NdArray(
                    data=fused_feature['voxel_coords'].tobytes(),
                    dtype=str(fused_feature['voxel_coords'].dtype),
                    shape=list(fused_feature['voxel_coords'].shape)
                ),
                voxel_num_points=Service_pb2.NdArray(
                    data=fused_feature['voxel_num_points'].tobytes(),
                    dtype=str(fused_feature['voxel_num_points'].dtype),
                    shape=list(fused_feature['voxel_num_points'].shape)
                )
            )
        )

    def GetFusedCommMask(self, request, context):  # 融合检测子系统向其他进程提供“获取融合后的协作图”的服务
        # ###################################################需要真实数据来源
        timestamp = int(time.time())  # 时间戳
        fused_comm_mask = np.array([34, 35, 36])  # 融合后的协作图
        # ###################################################

        return Service_pb2.FusedCommMask(  # 序列化并返回融合后的协作图
            timestamp=timestamp,
            fused_comm_mask=Service_pb2.NdArray(
                data=fused_comm_mask.tobytes(),
                dtype=str(fused_comm_mask.dtype),
                shape=list(fused_comm_mask.shape)
            )
        )

    def GetLatestPredBox(self, request, context):   # 融合检测子系统向其他进程提供“获取最新检测框”的服务
        # ###################################################需要真实数据来源
        timestamp = int(time.time())                        # 时间戳
        pred_box = self.pred_box.get_pred_box_copy()        # 最新检测框
        # ###################################################

        return Service_pb2.PredBox(  # 序列化并返回最新检测框
            timestamp=timestamp,
            pred_box=Service_pb2.NdArray(
                data=pred_box.tobytes(),
                dtype=str(pred_box.dtype),
                shape=list(pred_box.shape)
            )
        )

    def PCD2FeatureAndConfMap(self, request, context):
        timestamp = request.timestamp

        pcd = np.frombuffer(request.pcd.data, dtype=request.pcd.dtype).reshape(request.pcd.shape)

        feature = pcd2feature(pcd, self.hypes)

        conf_map = feature2conf_map(feature, self.model, self.device)

        return Service_pb2.FeatureAndConfMap(
            timestamp=timestamp,
            feature=Service_pb2.Feature(
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
            ),
            conf_map=Service_pb2.NdArray(
                data=conf_map.tobytes(),
                dtype=str(conf_map.dtype),
                shape=list(conf_map.shape)
            )
        )

    def Feature2ConfMap(self, request, context):
        timestamp = request.timestamp

        # 体素特征
        voxel_features_message = request.feature.voxel_features
        voxel_features = np.frombuffer(voxel_features_message.data,
                                       dtype=voxel_features_message.dtype).reshape(voxel_features_message.shape)
        # 体素坐标
        voxel_coords_message = request.feature.voxel_coords
        voxel_coords = np.frombuffer(voxel_coords_message.data,
                                     dtype=voxel_coords_message.dtype).reshape(voxel_coords_message.shape)

        # 体素点数
        voxel_num_points_message = request.feature.voxel_num_points
        voxel_num_points = np.frombuffer(voxel_num_points_message.data,
                                         dtype=voxel_num_points_message.dtype).reshape(voxel_num_points_message.shape)
        # 特征
        feature = {'voxel_features': voxel_features,
                   'voxel_coords': voxel_coords,
                   'voxel_num_points': voxel_num_points}

        conf_map = feature2conf_map(feature, self.model, self.device)

        return Service_pb2.ConfMap(
            timestamp=timestamp,
            conf_map=Service_pb2.NdArray(
                data=conf_map.tobytes(),
                dtype=str(conf_map.dtype),
                shape=list(conf_map.shape)
            )
        )

    def Feature2PredBox(self, request, context):
        timestamp = request.timestamp

        # 体素特征
        voxel_features_message = request.feature.voxel_features
        voxel_features = np.frombuffer(voxel_features_message.data,
                                       dtype=voxel_features_message.dtype).reshape(voxel_features_message.shape)
        # 体素坐标
        voxel_coords_message = request.feature.voxel_coords
        voxel_coords = np.frombuffer(voxel_coords_message.data,
                                     dtype=voxel_coords_message.dtype).reshape(voxel_coords_message.shape)

        # 体素点数
        voxel_num_points_message = request.feature.voxel_num_points
        voxel_num_points = np.frombuffer(voxel_num_points_message.data,
                                         dtype=voxel_num_points_message.dtype).reshape(voxel_num_points_message.shape)
        # 特征
        feature = {'voxel_features': voxel_features,
                   'voxel_coords': voxel_coords,
                   'voxel_num_points': voxel_num_points}
        pass


class DetectionServerThread(threading.Thread):
    def __init__(self, model, device, hypes, pred_box):
        super().__init__()
        self.model = model
        self.device = device
        self.hypes = hypes
        self.pred_box = pred_box

    def run(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
            ('grpc.max_send_message_length', 64 * 1024 * 1024),                         # 设置gRPC 消息的最大发送和接收大小为64MB
            ('grpc.max_receive_message_length', 64 * 1024 * 1024)])
        Service_pb2_grpc.add_DetectionServiceServicer_to_server(DetectionService(self.model, self.device, self.hypes, self.pred_box), server)
        server.add_insecure_port('[::]:50053')
        server.start()  # 非阻塞, 会实例化一个新线程来处理请求
        print("Detection Server is up and running on port 50053.")
        try:
            server.wait_for_termination()  # 保持服务器运行直到终止
        except KeyboardInterrupt:
            server.stop(0)  # 服务器终止
            print("Detection Server terminated.")


class DetectionClient:
    def __init__(self):
        perception_channel = grpc.insecure_channel('localhost:50051', options=[     # 与感知子系统建立连接
            ('grpc.max_send_message_length', 64 * 1024 * 1024),                     # 设置gRPC 消息的最大发送和接收大小为64MB
            ('grpc.max_receive_message_length', 64 * 1024 * 1024)])
        self.__perception_stub = Service_pb2_grpc.PerceptionServiceStub(perception_channel)

        collaboration_channel = grpc.insecure_channel('localhost:50052', options=[  # 与协同感知子系统建立连接
            ('grpc.max_send_message_length', 64 * 1024 * 1024),                     # 设置gRPC 消息的最大发送和接收大小为64MB
            ('grpc.max_receive_message_length', 64 * 1024 * 1024)])
        self.__collaboration_stub = Service_pb2_grpc.CollaborationServiceStub(collaboration_channel)

    def get_my_feature(self):  # 从感知子系统获取自车特征
        try:
            response = self.__perception_stub.GetMyFeature(Service_pb2.Empty(), timeout=5)  # 请求感知子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code}, details={e.details}")  # 记录grpc异常
            return -1, -1

        timestamp = response.timestamp  # 时间戳
        # 体素特征
        voxel_features_message = response.my_feature.voxel_features
        voxel_features = np.frombuffer(voxel_features_message.data,
                                       dtype=voxel_features_message.dtype).reshape(voxel_features_message.shape)
        # 体素坐标
        voxel_coords_message = response.my_feature.voxel_coords
        voxel_coords = np.frombuffer(voxel_coords_message.data,
                                     dtype=voxel_coords_message.dtype).reshape(voxel_coords_message.shape)

        # 体素点数
        voxel_num_points_message = response.my_feature.voxel_num_points
        voxel_num_points = np.frombuffer(voxel_num_points_message.data,
                                         dtype=voxel_num_points_message.dtype).reshape(voxel_num_points_message.shape)
        # 自车特征
        my_feature = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points}
        return timestamp, my_feature

    def get_others_info(self):  # 从协同感知子系统获取所有他车信息
        try:
            response = self.__collaboration_stub.GetOthersInfo(Service_pb2.Empty(), timeout=10)  # 请求协同感知子系统并获得响应
        except grpc.RpcError as e:  # 捕获grpc异常
            logging.error(f"RPC failed: code={e.code}, details={e.details}")  # 记录grpc异常
            return -1, -1, -1, -1, -1, -1, -1, -1

        ids = response.ids  # 所有他车的id
        timestamps = response.timestamps  # 所有他车传递信息对应的时间戳
        # 所有他车的位置
        poses = np.frombuffer(response.poses.data,
                              dtype=response.poses.dtype).reshape(response.poses.shape)
        # 所有他车的速度
        velocities = np.frombuffer(response.velocities.data,
                                   dtype=response.velocities.dtype).reshape(response.velocities.shape)
        # 所有他车的加速度
        accelerations = np.frombuffer(response.accelerations.data,
                                      dtype=response.accelerations.dtype).reshape(response.accelerations.shape)
        # 所有他车的体素特征
        voxel_features = np.frombuffer(response.voxel_features.data,
                                       dtype=response.voxel_features.dtype).reshape(response.voxel_features.shape)
        # 所有他车的体素坐标
        voxel_coords = np.frombuffer(response.voxel_coords.data,
                                     dtype=response.voxel_coords.dtype).reshape(response.voxel_coords.shape)
        # 所有他车的体素点数
        voxel_num_points = np.frombuffer(response.voxel_num_points.data,
                                         dtype=response.voxel_num_points.dtype).reshape(response.voxel_num_points.shape)

        return ids, timestamps, poses, velocities, accelerations, voxel_features, voxel_coords, voxel_num_points
