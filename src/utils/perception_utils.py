import logging

import numpy as np
import yaml.scanner

from utils.common import load_yaml, load_json


def get_lidar_pose_and_pcd_from_dataset(path):
    import open3d as o3d

    path = path.split('.')[0]
    try:
        yaml_load = load_yaml(path + '.yaml')
        lidar_pose = np.asarray(yaml_load['lidar_pose'])
    except FileNotFoundError as e:
        logging.error(e)
        lidar_pose = np.array([])
    except TypeError:
        logging.error(f'YAML file is empty or contains malformed content. File: \'{path}.yaml\'')
        lidar_pose = np.array([])
    except yaml.scanner.ScannerError:
        logging.error(f'YAML file contains malformed content. File: \'{path}.yaml\'')
        lidar_pose = np.array([])
    except KeyError as e:
        logging.error(f'Key {e} not found in YAML file. File: \'{path}.yaml\'')
        lidar_pose = np.array([])

    pcd_load = o3d.io.read_point_cloud(path + '.pcd')

    # 将Open3D的点云对象转换为NumPy数组
    xyz = np.asarray(pcd_load.points)
    intensity = np.expand_dims(np.asarray(pcd_load.colors)[:, 0], -1)
    pcd = np.hstack((xyz, intensity))
    return lidar_pose, pcd


def get_psa_from_obu(path):
    try:
        json_load = load_json(path)
        if all(key in json_load for key in {'pos_valid', 'lon', 'lat', 'ele', 'hea'}) and json_load['pos_valid']:
            lidar_pose = np.array([json_load['lon'], json_load['lat'], json_load['ele'], 0.0, json_load['hea'], 0.0])
        else:
            lidar_pose = np.array([])
        speed = json_load.get('spd', 0.0)
        acceleration = json_load.get('acceleration', 0.0)
    except Exception as e:
        logging.error(f'Json file load error. File: \'{path}\'')
        return np.array([]), 0.0, 0.0

    return lidar_pose, speed, acceleration


def save_lidar_pose_and_pcd(lidar_pose, pcd, path='../pcds/', file_name='pcd1'):
    file_name = file_name.split('.')[0]
    file_path = path + file_name + '.json'
    data = {'lidar_pose': lidar_pose.tolist(), 'pcd': pcd.tolist()}

    import os
    try:
        os.mkdir(path)
    except Exception:
        pass

    import json
    with open(file_path, 'w', newline='') as f:
        json.dump(data, f, indent=2)







