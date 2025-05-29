import logging
import os
import time
import threading
from appConfig import AppConfig

# from opencood.utils.pcd_utils import pcd_to_np
# from opencood.hypes_yaml.yaml_utils import load_yaml
import yaml.scanner
import concurrent.futures
from perception.perceptionRPCServer import PerceptionServerThread
from utils.sharedInfo import SharedInfo
from utils.common import load_yaml
import numpy as np
import queue

# def load_pose_and_pcd(path):
#    yaml_load = load_yaml(path + '.yaml')
#    pose = np.asarray(yaml_load['lidar_pose'])
#    pcd = pcd_to_np(path + '.pcd')
#    return pose, pcd

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

class ROSWrapper:
    def __init__(self, pcd_queue: queue.Queue):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        rospy.init_node("cosense_ros_node", disable_signals=True)
        self.pcd_queue = pcd_queue

    def _ros_init(self):
        self.sub = rospy.Subscriber(
            AppConfig.ros_pcd_topic,
            PointCloud2,
            self.pointcloud_callback,
            queue_size=1
        )
        rospy.spin()

    def pointcloud_callback(self, msg):
        try:
            points = point_cloud2.read_points_list(msg, field_names=("x", "y", "z", "intensity"))
            pcd = np.array(points)
            self.pcd_queue.put(pcd)
            logging.info(f"收到点云数据: {len(points)} 个点")
        except Exception as e:
            logging.error(f"点云处理失败: {str(e)}")

    def start(self):
        logging.info("启动ROS线程")
        self.executor.submit(self._ros_init)

class PerceptionManager:
    def __init__(self, cfg: AppConfig):
        self.my_info = SharedInfo()
        self.running = False
        self.cfg = cfg
        self.pcd_queue = queue.Queue(1)
        self.perception_rpc_server = PerceptionServerThread(self.my_info)
        self.ros_thread = threading.Thread(target=self.ros_start)

    def start(self):
        self.running = True
        self.ros_thread.start()
        self.perception_rpc_server.start()
        logging.debug('roswrapper start')
        self.__loop()

 #   def retrieve_from_dataset(self, index):
 #       paths = os.listdir(self.cfg.static_asset_path)
 #       pose, pcd = load_pose_and_pcd(paths[index % len(paths)])
 #       self.my_info.update_perception_info(pose=pose, pcd=pcd)

    def retrieve_from_ros(self, index):
        pcd = self.pcd_queue.get()
        logging.info(f'received pcd {len(pcd)}')
        self.my_info.update_perception_info(pcd=pcd)
    
    def ros_start(self):
    	self.ros_wrapper = ROSWrapper(self.pcd_queue)
    	self.ros_wrapper.start()

    def __loop(self):
        loop_time = 6
        last_t = 0
        loop_index = 0
        while self.running:
            t = time.time()
            if t - last_t < loop_time:
                time.sleep(loop_time + last_t - t)
            last_t = time.time()
            #if self.cfg.perception_debug:
             #   self.retrieve_from_dataset(loop_index)
            #else:
            self.retrieve_from_ros(loop_index)

            loop_index += 1

    def close(self):
        self.running = False
        self.perception_rpc_server.close()
