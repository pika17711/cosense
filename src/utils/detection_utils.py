import torch
import numpy as np

from collections import OrderedDict
from opencood.utils.pcd_utils import mask_points_by_range, mask_ego_points, shuffle_points
from opencood.utils.transformation_utils import x1_to_x2, gps_to_utm_transformation, gps_to_enu_transformation
from opencood.utils import box_utils
from opencood.tools import train_utils


def process_pcd(pcd, hypes):        # 对pcd点云数据进行预处理
    # pcd = shuffle_points(pcd)                 # 打乱数据
    # pcd = mask_ego_points(pcd)      # 去除打在自车上的点云
    processed_pcd = mask_points_by_range(pcd, hypes['preprocess']['cav_lidar_range']) # 去除指定范围外的点云
    return processed_pcd


def processed_pcd_to_voxel(processed_pcd, shared_info):
    pre_processor = shared_info.get_pre_processor()
    with shared_info.pre_processor_lock:
        voxel = pre_processor.preprocess(processed_pcd)
    return voxel


def pcd_to_voxel(pcd, shared_info):  # 根据pcd点云数据获取特征
    processed_pcd = process_pcd(pcd, shared_info.get_hypes())

    voxel = processed_pcd_to_voxel(processed_pcd, shared_info)

    return processed_pcd, voxel


def voxel_to_spatial_feature(voxel, shared_info):
    model = shared_info.get_model()
    device = shared_info.get_device()

    voxel_features = torch.from_numpy(voxel['voxel_features'])

    voxel_coords = np.pad(voxel['voxel_coords'], ((0, 0), (1, 0)), mode='constant', constant_values=0)
    voxel_coords = torch.from_numpy(voxel_coords)

    voxel_num_points = torch.from_numpy(voxel['voxel_num_points'])

    record_len = torch.empty(1, dtype=torch.int32)

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

    spatial_feature = batch_dict['spatial_features']
    return spatial_feature


def processed_pcd_to_spatial_feature(processed_pcd, shared_info):
    voxel = processed_pcd_to_voxel(processed_pcd, shared_info)
    spatial_feature_tensor = voxel_to_spatial_feature(voxel, shared_info)
    spatial_feature = spatial_feature_tensor.cpu().data.numpy()
    return spatial_feature


def pcd_to_spatial_feature(pcd, shared_info):
    processed_pcd = process_pcd(pcd, shared_info.get_hypes())
    spatial_feature = processed_pcd_to_spatial_feature(processed_pcd, shared_info)
    return processed_pcd, spatial_feature


def spatial_feature_to_comm_masked_feature(spatial_feature, shared_info, request_map=None):
    model = shared_info.get_model()
    device = shared_info.get_device()

    if isinstance(spatial_feature, np.ndarray):
        spatial_feature = torch.from_numpy(spatial_feature).to(device)

    if request_map is not None:
        request_map = torch.from_numpy(request_map.copy()).to(device)

    record_len = torch.tensor([spatial_feature.shape[0]], dtype=torch.int32).to(device)
    pairwise_t_matrix = torch.zeros((1, 5, 5, 4, 4), dtype=torch.float64).to(device)

    with shared_info.model_lock:
        with torch.no_grad():
            spatial_features_2d = model.backbone({'spatial_features': spatial_feature})['spatial_features_2d']

            if model.shrink_flag:
                spatial_features_2d = model.shrink_conv(spatial_features_2d)

            psm_single = model.cls_head(spatial_features_2d)

            if model.compression:
                # The ego feature is also compressed
                spatial_features_2d = model.naive_compressor(spatial_features_2d)

            if model.multi_scale:
                # Bypass communication cost, communicate at high resolution, neither shrink nor compress
                comm_masked_feature_tensor, comm_mask_tensor = model.fusion_net.spatial_feature_to_comm_masked_feature(spatial_feature,
                                                                      psm_single,
                                                                      record_len,
                                                                      pairwise_t_matrix,
                                                                      model.backbone,
                                                                      request_map=request_map)
            else:
                comm_masked_feature_tensor, comm_mask_tensor = model.fusion_net.spatial_feature_to_comm_masked_feature(spatial_features_2d,
                                                                      psm_single,
                                                                      record_len,
                                                                      pairwise_t_matrix,
                                                                      request_map=request_map)
    if comm_masked_feature_tensor is not None:
        comm_masked_feature = comm_masked_feature_tensor.cpu().data.numpy()
    else:
        comm_masked_feature = np.array([])

    if comm_mask_tensor is not None:
        comm_mask = comm_mask_tensor.cpu().data.numpy()
    else:
        comm_mask = np.array([])

    return comm_masked_feature, comm_mask


def process_spatial_feature(spatial_feature, shared_info, comm_masked_features=None):
    model = shared_info.get_model()
    device = shared_info.get_device()

    if isinstance(spatial_feature, np.ndarray):
        spatial_feature = torch.from_numpy(spatial_feature).to(device)

    if comm_masked_features is not None:
        for comm_masked_feature in comm_masked_features:
            for key, value in comm_masked_feature.items():
                comm_masked_feature[key] = torch.from_numpy(value.copy()).to(device)

    record_len = torch.tensor([spatial_feature.shape[0]], dtype=torch.int32).to(device)
    pairwise_t_matrix = torch.zeros((1, 5, 5, 4, 4), dtype=torch.float64).to(device)

    with shared_info.model_lock:
        with torch.no_grad():
            spatial_features_2d = model.backbone({'spatial_features': spatial_feature})['spatial_features_2d']

            if model.shrink_flag:
                spatial_features_2d = model.shrink_conv(spatial_features_2d)

            psm_single = model.cls_head(spatial_features_2d)

            if model.compression:
                # The ego feature is also compressed
                spatial_features_2d = model.naive_compressor(spatial_features_2d)

            if model.multi_scale:
                # Bypass communication cost, communicate at high resolution, neither shrink nor compress
                fused_feature, communication_rates, ego_comm_mask_tensor = model.fusion_net(spatial_feature,
                                                                                     psm_single,
                                                                                     record_len,
                                                                                     pairwise_t_matrix,
                                                                                     model.backbone,
                                                                                     comm_masked_features)
                if model.shrink_flag:
                    fused_feature = model.shrink_conv(fused_feature)
            else:
                fused_feature, communication_rates, ego_comm_mask_tensor = model.fusion_net(spatial_features_2d,
                                                                                     psm_single,
                                                                                     record_len,
                                                                                     pairwise_t_matrix,
                                                                                     comm_masked_features)

            psm = model.cls_head(fused_feature)
            rm = model.reg_head(fused_feature)

    output_dict = {'psm': psm, 'rm': rm, 'com': communication_rates}
    conf_map_tensor = 0
    conf_map = conf_map_tensor

    ego_comm_mask = ego_comm_mask_tensor.cpu().data.numpy()

    return output_dict, ego_comm_mask, conf_map


def spatial_feature_to_conf_map(spatial_feature, shared_info):  # 根据特征获取置信图
    _, _, conf_map = process_spatial_feature(spatial_feature, shared_info)
    return conf_map


def spatial_feature_to_comm_mask(spatial_feature, shared_info):  # 根据特征获取置信图
    _, comm_mask, _ = process_spatial_feature(spatial_feature, shared_info)
    return comm_mask


def spatial_feature_to_pred_box(spatial_feature, shared_info, comm_masked_features=None):      # 根据特征获取检测框
    output_dict = OrderedDict()
    output_dict['ego'], ego_comm_mask, _ = process_spatial_feature(spatial_feature, shared_info, comm_masked_features)

    device = shared_info.get_device()
    post_processor = shared_info.get_post_processor()

    transformation_matrix = np.eye(4, dtype=np.float32)
    transformation_matrix = torch.from_numpy(transformation_matrix)

    with shared_info.post_processor_lock:
        anchor_box = post_processor.generate_anchor_box()
    anchor_box = torch.from_numpy(anchor_box)

    batch_data = {'ego': {
        'transformation_matrix': transformation_matrix,
        'anchor_box': anchor_box
    }}

    with torch.no_grad():
        batch_data = train_utils.to_device(batch_data, device)
    with shared_info.post_processor_lock:
        pred_box_tensor, _ = post_processor.post_process(batch_data, output_dict)

    if pred_box_tensor is not None:
        pred_box = pred_box_tensor.cpu().data.numpy()
    else:
        pred_box = np.array([])

    return pred_box, ego_comm_mask


def voxel_to_conf_map(voxel, shared_info):  # 根据特征获取置信图
    spatial_feature = voxel_to_spatial_feature(voxel, shared_info)
    conf_map = spatial_feature_to_conf_map(spatial_feature, shared_info)
    return conf_map


def voxel_to_pred_box(voxel, shared_info):      # 根据特征获取检测框
    spatial_feature = voxel_to_spatial_feature(voxel, shared_info)
    pred_box, _ = spatial_feature_to_pred_box(spatial_feature, shared_info)

    return pred_box


def project_pcd(my_pcd, my_lidar_pose, other_lidar_pose, hypes, gps=False):
    projected_pcd = my_pcd.copy()

    # projected_pcd = shuffle_points(projected_pcd)
    projected_pcd = mask_ego_points(projected_pcd)

    if gps:
        transformation_matrix = gps_to_utm_transformation(my_lidar_pose, other_lidar_pose)
    else:
        transformation_matrix = x1_to_x2(my_lidar_pose, other_lidar_pose)

    projected_pcd[:, :3] = box_utils.project_points_by_matrix_torch(projected_pcd[:, :3], transformation_matrix)
    projected_pcd = mask_points_by_range(projected_pcd, hypes['preprocess']['cav_lidar_range'])

    return projected_pcd


def lidar_poses_to_projected_voxel(my_lidar_pose, my_pcd, lidar_poses, shared_info, gps=False):
    hypes = shared_info.get_hypes()

    voxels_lens = []
    voxels_features = []
    voxels_coords = []
    voxels_num_points = []
    for lidar_pose in lidar_poses:
        projected_pcd = project_pcd(my_pcd, my_lidar_pose, lidar_pose, hypes, gps)
        projected_voxel = processed_pcd_to_voxel(projected_pcd, shared_info)
        voxels_lens.append(projected_voxel['voxel_features'].shape[0])
        voxels_features.append(projected_voxel['voxel_features'])
        voxels_coords.append(projected_voxel['voxel_coords'])
        voxels_num_points.append(projected_voxel['voxel_num_points'])
    voxel_features = np.vstack(voxels_features)
    voxel_coords = np.vstack(voxels_coords)
    voxel_num_points = np.hstack(voxels_num_points)

    projected_voxels = {
        'voxel_lens': voxels_lens,
        'voxel_features': voxel_features,
        'voxel_coords': voxel_coords,
        'voxel_num_points': voxel_num_points
    }
    return projected_voxels

def lidar_pose_to_projected_spatial_feature(my_lidar_pose, my_pcd, lidar_pose, shared_info, gps=False):
    hypes = shared_info.get_hypes()

    projected_pcd = project_pcd(my_pcd, my_lidar_pose, lidar_pose, hypes, gps)
    projected_spatial_feature = processed_pcd_to_spatial_feature(projected_pcd, shared_info)

    return projected_spatial_feature

def lidar_poses_to_projected_spatial_features(my_lidar_pose, my_pcd, lidar_poses, shared_info, gps=False):
    hypes = shared_info.get_hypes()

    projected_spatial_features = {}
    for cav_id, lidar_pose in lidar_poses.items():
        projected_spatial_feature = lidar_pose_to_projected_spatial_feature(my_lidar_pose, my_pcd, lidar_pose['lidar_pose'], shared_info, gps)
        projected_spatial_features[cav_id] = {'feature': projected_spatial_feature,
                                              'ts_feature': lidar_pose['ts_lidar_pose']}
    return projected_spatial_features


def request_map_to_comm_masked_feature(my_lidar_pose, my_pcd, target_lidar_pose, target_request_map, shared_info, gps=False):
    projected_spatial_feature = lidar_pose_to_projected_spatial_feature(my_lidar_pose, my_pcd, target_lidar_pose, shared_info, gps)
    comm_masked_feature, comm_mask = spatial_feature_to_comm_masked_feature(projected_spatial_feature, shared_info, target_request_map)
    return comm_masked_feature, comm_mask


def fuse_pcd(my_timestamp, my_pcd, my_lidar_pose, timestamps, others_lidar_poses, others_pcds):
    fused_pcd = [my_pcd]

    for i in range(len(timestamps)):
        # if timestamps[i] == my_timestamp:
        #     transformation_matrix = x1_to_x2(others_lidar_poses[i], my_lidar_pose)
        #     others_pcds[:, :3] = box_utils.project_points_by_matrix_torch(others_pcds[:, :3], transformation_matrix)
        #     fused_pcd.append(others_pcds)
        transformation_matrix = x1_to_x2(others_lidar_poses[i], my_lidar_pose)
        others_pcd = others_pcds[i].copy()
        others_pcd[:, :3] = box_utils.project_points_by_matrix_torch(others_pcd[:, :3], transformation_matrix)
        fused_pcd.append(others_pcd)

    fused_pcd = np.vstack(fused_pcd)
    return fused_pcd


def fuse_voxel(my_feature, voxel_features, voxel_coords, voxel_num_points):
    fused_voxel_features = [my_feature['voxel_features']]
    fused_voxel_coords = [my_feature['voxel_coords']]
    fused_voxel_num_points = [my_feature['voxel_num_points']]

    if voxel_features.shape[0] > 0:
        fused_voxel_features.append(voxel_features)
        fused_voxel_coords.append(voxel_coords)
        fused_voxel_num_points.append(voxel_num_points)

    fused_voxel_features = np.vstack(fused_voxel_features)
    fused_voxel_coords = np.vstack(fused_voxel_coords)
    fused_voxel_num_points = np.hstack(fused_voxel_num_points)

    fused_voxel = {'voxel_features': fused_voxel_features,
                   'voxel_coords': fused_voxel_coords,
                   'voxel_num_points': fused_voxel_num_points}

    return fused_voxel


def get_features_from_cav_infos(cav_infos):
    spatial_features = []
    comm_masked_features = []

    if cav_infos is not None:
        for cav_id, cav_info in cav_infos.items():
            if 'comm_mask' not in cav_info or cav_info['comm_mask'] is None:
                spatial_features.append(cav_info['feature'])
            else:
                comm_masked_features.append({'comm_masked_feature': cav_info['feature'],
                                             'comm_mask': cav_info['comm_mask']})

    return spatial_features, comm_masked_features


def fuse_spatial_feature(my_spatial_feature, spatial_features):
    spatial_features.insert(0, my_spatial_feature)
    fused_spatial_feature = np.vstack(spatial_features)
    return fused_spatial_feature
