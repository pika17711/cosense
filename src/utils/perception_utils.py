import logging
import pickle
import ros_numpy

import numpy as np
import yaml.scanner

from utils.common import load_yaml, load_json
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2


def ros_pcd_to_numpy(ros_pcd: PointCloud2) -> np.array:
    # points = point_cloud2.read_points_list(ros_pcd, field_names=("x", "y", "z", "intensity"), skip_nans=True)
    # pcd_np = np.array(points)

    pcd_array = ros_numpy.numpify(ros_pcd)

    pcd_x = pcd_array[:, :]['x'].flatten()
    pcd_y = pcd_array[:, :]['y'].flatten()
    pcd_z = pcd_array[:, :]['z'].flatten()
    pcd_intensity = pcd_array[:, :]['intensity'].flatten() / 255.0

    pcd_np = np.vstack([pcd_x, pcd_y, pcd_z, pcd_intensity]).T

    coord_valid = ~np.isnan(pcd_np[:, :3]).any(axis=1)
    pcd_np = pcd_np[coord_valid]

    return pcd_np


def get_lidar_pose_and_pcd_from_dataset(file_path):
    file_type = file_path.split('.')[-1]

    lidar_pose = None
    pcd = None

    if file_type == 'pcd' or file_type == 'yaml':
        import open3d as o3d

        path = file_path.split('.')[0]
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
    elif file_type == 'json':
        json_load = load_json(file_path)

        lidar_pose = np.array(json_load['lidar_pose']) if 'lidar_pose' in json_load else None
        pcd = np.array(json_load['pcd']) if 'pcd' in json_load else None
        if isinstance(pcd, np.ndarray):
            pcd[:, 3] = pcd[:, 3] / 255.0
    elif file_type == 'txt':
        with open(file_path, 'rb') as file:
            binary_data = file.read()
        data_dict = pickle.loads(binary_data)

        lidar_pose = np.array(data_dict['lidar_pose']) if 'lidar_pose' in data_dict else None
        pcd = np.array(data_dict['pcd']) if 'pcd' in data_dict else None
        if isinstance(pcd, np.ndarray):
            pcd[:, 3] = pcd[:, 3] / 255.0

    return lidar_pose, pcd


def get_psa_from_obu(path):
    try:
        json_load = load_json(path)
        if all(key in json_load for key in {'pos_valid', 'lon', 'lat', 'ele', 'hea'}) and json_load['pos_valid']:
            lidar_pose = np.array([json_load['lon'], json_load['lat'], json_load['ele'], 0.0, 0.0, json_load['hea']])
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

    json_path = path + file_name + '.json'
    txt_path = path + file_name + '.txt'

    data_dict = {'lidar_pose': lidar_pose.tolist() if lidar_pose is not None else None,
                 'pcd': pcd.tolist() if pcd is not None else None}
    import pickle
    binary_data = pickle.dumps(data_dict)

    import os
    try:
        os.mkdir(path)
    except Exception:
        pass

    import json
    with open(json_path, 'w', newline='') as f:
        json.dump(data_dict, f, indent=2)

    with open(txt_path, 'wb') as file:
        file.write(binary_data)

    exit(0)