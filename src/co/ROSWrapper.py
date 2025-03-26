import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

import numpy as np
from server import CONFIG
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

from server import agent_server

class ROSWrapper:
    def __init__(self):
        self.thread = None
        self.executor = ThreadPoolExecutor(1)
        rospy.init_node("async_ros_node", disable_signals=True)
        self.logger = logging.getLogger('ROS')
        self.main_eventloop = None

    def _ros_init(self, loop):
        self.main_eventloop: asyncio.AbstractEventLoop = loop
        self.sub = rospy.Subscriber(
            CONFIG['ros']['pointcloud_topic'],
            PointCloud2,
            self.pointcloud_callback,
            queue_size=1
        )
        rospy.spin()

    def pointcloud_callback(self, msg):
        try:
            points = point_cloud2.read_points_list(msg, field_names=("x", "y", "z", "intensity"))
            pcd = np.array(points)
            self.main_eventloop.create_task(agent_server.set_ego_feat(pcd))
            self.logger.info(f"收到点云数据: {len(points)} 个点")
        except Exception as e:
            self.logger.error(f"点云处理失败: {str(e)}")

    async def run_in_thread(self):
        self.logger.info("启动ROS线程")
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self.executor, self._ros_init, loop)
        except Exception as e:
            self.logger.critical(f"ROS线程启动失败: {str(e)}")
            raise

