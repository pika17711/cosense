import concurrent.futures
import queue
import logging
import numpy as np
import ros_numpy

from appConfig import AppConfig
from utils.perception_utils import ros_pcd_to_numpy

import rospy
from collections import deque
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2


class ROSWrapper:
    def __init__(self, pcd_queue: deque):
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

    def pointcloud_callback(self, ros_pcd: PointCloud2):
        try:
            self.pcd_queue.append(ros_pcd)  # deque
        except Exception as e:
            logging.error(f"点云处理失败: {str(e)}")

    def start(self):
        logging.info("启动ROS线程")
        self.executor.submit(self._ros_init)
