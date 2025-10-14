import json
import time
import numpy as np

from appConfig import AppConfig
from collaboration.collaborationRPCServer import CollaborationRPCServerThread
from collaboration.collaborationTable import CollaborationTable
from collaboration.coopMap import CoopMap, CoopMapType
from perception.perceptionRPCClient import PerceptionRPCClient
from detection.detectionRPCClient import DetectionRPCClient
from utils.othersInfos import OthersInfos
from utils import InfoDTO


def write_info(my_id, ts_lidar_pose, my_lidar_pose, projected_features):
    info_path = 'D:\\WorkSpace\\Python\\Test\\json\\' + str(my_id) + '_info.json'

    projected_features_list = []

    for target_id, projected_feature in projected_features.items():
        projected_feature = {'target_id': target_id,
                             'ts_feature': projected_feature['ts_feature'],
                             'projected_feature': projected_feature['feature'].tolist()}
        projected_features_list.append(projected_feature)

    data = {'cav_id': my_id,
            'ts_lidar_pose': ts_lidar_pose,
            'lidar_pose': my_lidar_pose.tolist() if my_lidar_pose is not None else [],
            'projected_features': projected_features_list}

    with open(info_path, 'w', newline='') as f:
        json.dump(data, f, indent=2)


def read_info(other_id, my_id):
    info_path = 'D:\\WorkSpace\\Python\\Test\\json\\' + str(other_id) + '_info.json'

    try:
        with open(info_path, 'r', newline='') as f:
            data = json.load(f)
            if 'cav_id' not in data:
                return None, None
            cav_id = str(data['cav_id'])
            cav_info = {
                'ts_lidar_pose': data['ts_lidar_pose'] if 'ts_lidar_pose' in data else None,
                'lidar_pose': np.array(data['lidar_pose']) if 'lidar_pose' in data else None}

            if 'projected_features' in data:
                projected_features = data['projected_features']
                for projected_feature in projected_features:
                    if projected_feature['target_id'] == my_id:
                        cav_info['ts_feature'] = projected_feature['ts_feature']
                        cav_info['feature'] = np.array(projected_feature['projected_feature'], dtype=np.float32)


    except json.decoder.JSONDecodeError:
        return None, None

    return cav_id, cav_info


def write_coopmap(my_id, coopmap):
    coopmap_path = 'D:\\WorkSpace\\Python\\Test\\txt\\' + str(my_id) + '_coopmap.txt'

    decoopmap = CoopMap.serialize(coopmap)

    with open(coopmap_path, 'wb') as file:
        file.write(decoopmap)


def read_coopmap(other_id):
    coopmap_path = 'D:\\WorkSpace\\Python\\Test\\txt\\' + str(other_id) + '_coopmap.txt'
    try:
        with open(coopmap_path, 'rb') as file:
            decoopmap = file.read()
            coopmap = CoopMap.deserialize(decoopmap)

        return coopmap
    except Exception:
        return None


def write_data(my_id, data):
    data_path = 'D:\\WorkSpace\\Python\\Test\\txt\\' + str(my_id) + '_data.txt'

    with open(data_path, 'wb') as file:
        file.write(data)


def read_data(other_id):
    data_path = 'D:\\WorkSpace\\Python\\Test\\txt\\' + str(other_id) + '_data.txt'
    try:
        with open(data_path, 'rb') as file:
            data = file.read()

            de_data = InfoDTO.InfoDTOSerializer.deserialize(data)

        return de_data
    except Exception:
        return None


class CollaborationManager:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

        self.__my_id = '302'
        self.__others_ids = ['311']

        self.__ctable = CollaborationTable(cfg)

        self.others_infos = OthersInfos(self.__ctable)

        self.running = False
        self.__grpc_prepare()

    def __grpc_prepare(self):
        self.collaboration_rpc_server = CollaborationRPCServerThread(self.cfg, self.others_infos)
        self.collaboration_rpc_server.start()

        self.perception_client = PerceptionRPCClient(self.cfg)
        self.detection_client = DetectionRPCClient()

    def start(self):
        self.running = True
        self.__loop()

    def __get_others_info(self):
        # others_infos = {}

        # #######################################################################获取其他车辆的信息
        for cav_id in self.__others_ids:
            # cav_id, cav_info = read_info(other_id=cav_id, my_id=self.__my_id)
            data = read_data(cav_id)

            # others_infos[cav_id] = cav_info
            if data is not None:
                self.__ctable.add_data(data)
        # #######################################################################

        # self.others_infos.update_infos(others_infos)

    def __loop(self):
        loop_time = 6
        last_t = 0

        while True:
            t = time.time()
            if t - last_t < loop_time:
                time.sleep(loop_time + last_t - t)
            last_t = time.time()

            self.__get_others_info()

            my_coopmap = self.__get_self_coopmap(CoopMapType.RequestMap)
            # my_coopmap.map = my_coopmap.map.astype(bool)
            if my_coopmap is not None:
                write_coopmap(self.__my_id, my_coopmap)

            for cav_id in self.__others_ids:
                coopmap = read_coopmap(cav_id)
                # if self.cfg.collaboration_no_coopmap_debug:
                if coopmap is not None and coopmap.type == CoopMapType.RequestMap:
                    if self.cfg.collaboration_request_map_debug:
                        coopmap.map[:] = 1
                    data = self.__get_all_data(coopmap)
                    write_data(self.__my_id, data)

            # others_infos = self.others_infos.get_infos()

            # valid_lidar_poses = {}
            # for cav_id, cav_info in others_infos.items():
            #     if cav_info['lidar_pose'].size > 0:
            #         valid_lidar_poses[cav_id] = {'lidar_pose': cav_info['lidar_pose'],
            #                                      'ts_lidar_pose': cav_info['ts_lidar_pose']}
            #
            # if len(valid_lidar_poses) > 0:
            #     projected_spatial_features = \
            #         self.detection_client.lidar_poses_to_projected_spatial_features(valid_lidar_poses)
            # else:
            #     projected_spatial_features = {}
            #
            # my_lidar_pose, ts_lidar_pose, _, _, _, _ = self.perception_client.get_my_pva()
            #
            # write_info(self.__my_id, ts_lidar_pose, my_lidar_pose, projected_spatial_features)  # 传输自车的特征

    def __get_self_coopmap(self, coopmap_type: CoopMapType = CoopMapType.CommMask):
        if coopmap_type == CoopMapType.DEBUG:
            return CoopMap(self.cfg.id, coopmap_type, None, None)

        comm_mask, _, lidar_pose, _ = self.detection_client.get_comm_mask_and_lidar_pose()
        if comm_mask is None:
            return None

        if coopmap_type == CoopMapType.Empty:
            return CoopMap(self.cfg.id, coopmap_type, None, lidar_pose)
        if coopmap_type == CoopMapType.CommMask:
            return CoopMap(self.cfg.id, coopmap_type, comm_mask, lidar_pose)
        if coopmap_type == CoopMapType.RequestMap:
            request_map = 1 - comm_mask
            return CoopMap(self.cfg.id, coopmap_type, request_map, lidar_pose)

    def __get_all_data(self, coopmap: CoopMap):
        lidar_pose, ts_lidar_pose, speed, ts_spd, acceleration, ts_acc = self.perception_client.get_my_pva()
        # my_extrinsic_matrix, ts_extrinsic_matrix = self.perception_client.get_my_extrinsic_matrix()
        # lidar_poses = {'other': {'lidar_pose': coopmap.lidar_pose, 'ts_lidar_pose': int(time.time())}}
        # projected_spatial_feature = self.detection_client.lidar_poses_to_projected_spatial_features(lidar_poses)

        #####################
        test_lidar_pose = coopmap.lidar_pose.copy()
        test_lidar_pose[2] = test_lidar_pose[2] - 100
        #####################
        comm_masked_feature, comm_mask, ts_feature = self.detection_client.request_map_to_projected_comm_masked_feature(
            test_lidar_pose, coopmap.map, int(time.time()))
        if self.cfg.collaboration_pcd_debug:
            pcd, ts_pcd = self.perception_client.get_my_pcd()
        else:
            pcd = None
            ts_pcd = None

        infodto = InfoDTO.InfoDTO(type=1,
                                  id=self.cfg.id,
                                  lidar2world=None,
                                  camera2world=None,
                                  camera_intrinsic=None,
                                  feat={'spatial_feature': comm_masked_feature,
                                        'comm_mask': comm_mask},
                                  ts_feat=ts_feature,
                                  speed=speed,
                                  ts_spd=ts_spd,
                                  lidar_pos=lidar_pose,
                                  ts_lidar_pos=ts_lidar_pose,
                                  acc=acceleration,
                                  ts_acc=ts_acc,
                                  pcd=pcd,
                                  ts_pcd=ts_pcd)
        data = InfoDTO.InfoDTOSerializer.serialize(infodto)
        # data = InfoDTO.InfoDTOSerializer.serialize_to_str(infodto)
        return data

    def close(self):
        self.running = False
        self.collaboration_rpc_server.close()
