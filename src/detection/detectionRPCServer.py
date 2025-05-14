import threading
import grpc
import time
import torch
import numpy as np
from concurrent import futures
from collections import OrderedDict

from rpc import Service_pb2_grpc
from rpc import Service_pb2
from opencood.utils.pcd_utils import mask_points_by_range, mask_ego_points, shuffle_points
from opencood.utils.transformation_utils import x1_to_x2
from opencood.utils import box_utils
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.tools import train_utils


def process_pcd(pcd, hypes):        # 对pcd点云数据进行预处理
    processed_pcd = shuffle_points(pcd)                 # 打乱数据
    processed_pcd = mask_ego_points(processed_pcd)      # 去除打在自车上的点云
    processed_pcd = mask_points_by_range(processed_pcd, hypes['preprocess']['cav_lidar_range']) # 去除指定范围外的点云
    return processed_pcd


def pcd2feature(pcd, shared_info):  # 根据pcd点云数据获取特征
    processed_pcd = process_pcd(pcd, shared_info.get_hypes())

    with shared_info.pre_processor_lock:
        feature = shared_info.get_pre_processor().preprocess(processed_pcd)
    return processed_pcd, feature

def poses_to_projected_features(my_pose, my_pcd, poses, shared_info):
    features_lens = []
    voxel_features = []
    voxel_coords = []
    voxel_num_points = []
    for pose in poses:
        transformation_matrix = x1_to_x2(pose, my_pose)
        projected_pcd = my_pcd.copy()
        projected_pcd[:, :3] = box_utils.project_points_by_matrix_torch(projected_pcd[:, :3], transformation_matrix)
        _, projected_feature = pcd2feature(projected_pcd, shared_info)
        features_lens.append(projected_feature['voxel_features'].shape[0])
        voxel_features.append(projected_feature['voxel_features'])
        voxel_coords.append(projected_feature['voxel_coords'])
        voxel_num_points.append(projected_feature['voxel_num_points'])
    voxel_features = np.vstack(voxel_features)
    voxel_coords = np.vstack(voxel_coords)
    voxel_num_points = np.hstack(voxel_num_points)

    projected_features = {
        'features_lens': features_lens,
        'voxel_features': voxel_features,
        'voxel_coords': voxel_coords,
        'voxel_num_points' : voxel_num_points
    }
    return projected_features


def model_forward(feature, shared_info):              # 模型推理获取中间变量
    model = shared_info.get_model()
    device = shared_info.get_device()

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

    with shared_info.model_lock:
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

        # if model.multi_scale:
        #     # Bypass communication cost, communicate at high resolution, neither shrink nor compress
        #     fused_feature, communication_rates, conf_map_tensor = model.fusion_net(batch_dict['spatial_features'],
        #                                                                     psm_single,
        #                                                                     record_len,
        #                                                                     pairwise_t_matrix,
        #                                                                     model.backbone)
        #     if model.shrink_flag:
        #         fused_feature = model.shrink_conv(fused_feature)
        # else:
        #     fused_feature, communication_rates, conf_map_tensor = model.fusion_net(spatial_features_2d,
        #                                                                     psm_single,
        #                                                                     record_len,
        #                                                                     pairwise_t_matrix)
        if model.multi_scale:
            # Bypass communication cost, communicate at high resolution, neither shrink nor compress
            fused_feature, communication_rates = model.fusion_net(batch_dict['spatial_features'],
                                                                 psm_single,
                                                                 record_len,
                                                                 pairwise_t_matrix,
                                                                 model.backbone)
            if model.shrink_flag:
                fused_feature = model.shrink_conv(fused_feature)
        else:
            fused_feature, communication_rates = model.fusion_net(spatial_features_2d,
                                                                 psm_single,
                                                                 record_len,
                                                                 pairwise_t_matrix)

        psm = model.cls_head(fused_feature)
        rm = model.reg_head(fused_feature)

    output_dict = {'psm': psm, 'rm': rm, 'com': communication_rates}
    # return output_dict, conf_map_tensor
    return output_dict, 0


def feature2conf_map(feature, shared_info):  # 根据特征获取置信图
    _, conf_map_tensor = model_forward(feature, shared_info)
    conf_map = conf_map_tensor.cpu().data.numpy()
    return conf_map


def feature2pred_box(feature, shared_info):      # 根据特征获取检测框
    device = shared_info.get_device()
    dataset = shared_info.get_dataset()

    transformation_matrix = np.eye(4, dtype=np.float32)
    transformation_matrix = torch.from_numpy(transformation_matrix)

    with shared_info.dataset_lock:
        anchor_box = dataset.post_processor.generate_anchor_box()
    anchor_box = torch.from_numpy(anchor_box)

    batch_data = {'ego': {
        'transformation_matrix': transformation_matrix,
        'anchor_box': anchor_box
    }}

    with torch.no_grad():
        batch_data = train_utils.to_device(batch_data, device)

    output_dict = OrderedDict()
    output_dict['ego'], _ = model_forward(feature, shared_info)

    pred_box_tensor, _, _ = dataset.post_process(batch_data, output_dict)
    pred_box = pred_box_tensor.cpu().data.numpy()

    return pred_box


class DetectionRPCService(Service_pb2_grpc.DetectionServiceServicer):  # 融合检测子系统的Service类
    def __init__(self, shared_info):
        super().__init__()
        self.shared_info = shared_info

    def GetFusedFeature(self, request, context):  # 融合检测子系统向其他进程提供“获取融合后的特征”的服务
        timestamp = int(time.time())  # 时间戳
        fused_feature = self.shared_info.get_fused_feature_copy()

        return Service_pb2.Feature(  # 序列化并返回融合后的特征
            timestamp=timestamp,
            feature=Service_pb2._Feature(
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
        timestamp = int(time.time())  # 时间戳
        fused_comm_mask = self.shared_info.get_fused_comm_mask_copy()

        return Service_pb2.CommMask(  # 序列化并返回融合后的协作图
            timestamp=timestamp,
            comm_mask=Service_pb2.NdArray(
                data=fused_comm_mask.tobytes(),
                dtype=str(fused_comm_mask.dtype),
                shape=list(fused_comm_mask.shape)
            )
        )

    def GetLatestPredBox(self, request, context):  # 融合检测子系统向其他进程提供“获取最新检测框”的服务
        timestamp = int(time.time())                    # 时间戳
        pred_box = self.shared_info.get_pred_box_copy()    # 最新检测框

        return Service_pb2.PredBox(  # 序列化并返回最新检测框
            timestamp=timestamp,
            pred_box=Service_pb2.NdArray(
                data=pred_box.tobytes(),
                dtype=str(pred_box.dtype),
                shape=list(pred_box.shape)
            )
        )

    def PCD2Feature(self, request, context):
        pcd = np.frombuffer(request.pcd.data, dtype=request.pcd.dtype).reshape(request.pcd.shape)

        timestamp = request.timestamp
        # 特征
        _, feature = pcd2feature(pcd, self.shared_info)

        return Service_pb2.Feature(
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

    def Poses2ProjectedFeatures(self, request, context):
        timestamps = request.timestamps
        poses = np.frombuffer(request.poses.data, dtype=request.poses.dtype).reshape(request.poses.shape)
        # 特征
        my_pose = self.shared_info.get_pose_copy()
        my_pcd = self.shared_info.get_pcd_copy()
        projected_features = poses_to_projected_features(my_pose, my_pcd, poses, self.shared_info)

        return Service_pb2.Features(
            timestamps=timestamps,
            features_lens=projected_features['features_lens'],
            voxel_features=Service_pb2.NdArray(
                data=projected_features['voxel_features'].tobytes(),
                dtype=str(projected_features['voxel_features'].dtype),
                shape=list(projected_features['voxel_features'].shape)
            ),
            voxel_coords=Service_pb2.NdArray(
                data=projected_features['voxel_coords'].tobytes(),
                dtype=str(projected_features['voxel_coords'].dtype),
                shape=list(projected_features['voxel_coords'].shape)
            ),
            voxel_num_points=Service_pb2.NdArray(
                data=projected_features['voxel_num_points'].tobytes(),
                dtype=str(projected_features['voxel_num_points'].dtype),
                shape=list(projected_features['voxel_num_points'].shape)
            )
        )

    def PCD2FeatureAndConfMap(self, request, context):      # 融合检测子系统向其他进程提供“根据点云获取特征和置信图”的服务
        pcd = np.frombuffer(request.pcd.data, dtype=request.pcd.dtype).reshape(request.pcd.shape)

        timestamp = request.timestamp
        # 特征
        _, feature = pcd2feature(pcd, self.shared_info)
        # 置信图
        conf_map = feature2conf_map(feature, self.shared_info)

        return Service_pb2.FeatureAndConfMap(
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
            ),
            conf_map=Service_pb2.NdArray(
                data=conf_map.tobytes(),
                dtype=str(conf_map.dtype),
                shape=list(conf_map.shape)
            )
        )

    def Feature2ConfMap(self, request, context):            # 融合检测子系统向其他进程提供“根据特征获取置信图”的服务
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
        # 置信图
        conf_map = feature2conf_map(feature, self.shared_info)

        return Service_pb2.ConfMap(
            timestamp=timestamp,
            conf_map=Service_pb2.NdArray(
                data=conf_map.tobytes(),
                dtype=str(conf_map.dtype),
                shape=list(conf_map.shape)
            )
        )

    def Feature2PredBox(self, request, context):        # 融合检测子系统向其他进程提供“根据特征获取检测框”的服务
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
        # 检测框
        pred_box = feature2pred_box(feature, self.shared_info)

        return Service_pb2.PredBox(
            timestamp=timestamp,
            pred_box=Service_pb2.NdArray(
                data=pred_box.tobytes(),
                dtype=str(pred_box.dtype),
                shape=list(pred_box.shape)
            )
        )


class DetectionServerThread(threading.Thread):  # 融合检测子系统的Server线程
    def __init__(self, shared_info):
        super().__init__()
        self.shared_info = shared_info

    def run(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
            ('grpc.max_send_message_length', 64 * 1024 * 1024),  # 设置gRPC 消息的最大发送和接收大小为64MB
            ('grpc.max_receive_message_length', 64 * 1024 * 1024)])
        Service_pb2_grpc.add_DetectionServiceServicer_to_server(
            DetectionRPCService(self.shared_info), server)
        server.add_insecure_port('[::]:50053')
        server.start()  # 非阻塞, 会实例化一个新线程来处理请求
        print("Detection Server is up and running on port 50053.")
        try:
            server.wait_for_termination()  # 保持服务器运行直到终止
        except KeyboardInterrupt:
            server.stop(0)  # 服务器终止
            print("Detection Server terminated.")
