#!/usr/bin/env python
import argparse
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
import signal
import sys

CONFIG = {
    "name": "/test_pointcloud"
}

class TestPointCloudPublisher:
    def __init__(self):
        # 设置信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # 延迟初始化直到实际运行时
        self.pub = None

    def signal_handler(self, sig, frame):
        print("\n检测到Ctrl+C，正在关闭节点...")
        if self.pub is not None:
            self.pub.unregister()
        sys.exit(0)

    def initialize_node(self):
        """延迟的节点初始化"""
        rospy.init_node("test_pcl_publisher", anonymous=True)
        self.pub = rospy.Publisher(
            CONFIG["name"],
            PointCloud2,
            queue_size=1
        )

        # 生成测试点云的配置
        self.num_points = 100  # 测试点数量
        self.step = 0  # 用于生成动态数据

    def create_test_pointcloud(self):
        """生成带有x,y,z,intensity字段的测试点云"""
        points = []
        
        # 生成两种测试模式（可通过注释切换）：
        
        # 模式1：固定点云（调试基础功能）
        # for i in range(self.num_points):
        #     x = i * 0.1
        #     y = 0
        #     z = 0
        #     intensity = i % 255
        #     points.append([x, y, z, intensity])
        
        # 模式2：动态移动点云（调试持续数据处理）
        for i in range(self.num_points):
            x = np.cos(self.step * 0.1 + i * 0.1)
            y = np.sin(self.step * 0.1 + i * 0.1)
            z = i * 0.01
            intensity = (self.step + i) % 255
            points.append([x, y, z, intensity])
        
        self.step += 1
        return np.array(points, dtype=np.float32)

    def publish(self):
        """构造并发布PointCloud2消息"""
        # 创建消息头
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "test_frame"
        
        # 创建字段描述
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1)
        ]
        
        # 转换numpy数组为bytes
        points = self.create_test_pointcloud()
        data = points.tobytes()
        
        # 构造完整消息
        pcl_msg = PointCloud2(
            header=header,
            height=1,
            width=points.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=16,  # 4个float32 * 4bytes = 16
            row_step=16 * points.shape[0],
            data=data
        )
        
        self.pub.publish(pcl_msg)
        rospy.loginfo(f"Published {points.shape[0]} points")

    def run(self):
        try:
            self.initialize_node()
            rate = rospy.Rate(1)

            while not rospy.is_shutdown():
                if self.pub.get_num_connections() > 0:
                    self.publish()
                rate.sleep()
                
        except rospy.ROSInterruptException:
            print("节点被中断")
        finally:
            if self.pub is not None:
                self.pub.unregister()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="程序配置设置")
    parser.add_argument('--topic_name', type=str, help="发布点云数据的topic name", default=CONFIG["name"])

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    if args.topic_name:
        CONFIG["name"] = args.topic_name

    publisher = TestPointCloudPublisher()
    publisher.run()