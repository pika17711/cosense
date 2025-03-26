import argparse
import asyncio
import json
import logging

import zmq
import zmq.asyncio

from config import CONFIG

# from logging.handlers import RotatingFileHandler

# 在parse_args之后添加日志配置
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # # 文件处理器（自动轮转）
    # file_handler = RotatingFileHandler(
    #     'app.log',
    #     maxBytes=10 * 1024 * 1024,  # 10MB
    #     backupCount=5
    # )
    # file_handler.setLevel(logging.DEBUG)
    # file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    # logger.addHandler(file_handler)

from ICPHandler import ICP_handler

from client import agent_client

from server import agent_server

from GRPCClient import grpc_client


from baseDataRetriever import BaseDataRetriever
from boundingBoxAnnotator import BoundingBoxAnnotator
from communicationManager import CommunicationManager

class CollaborationSystem:
    """感知系统总控模块"""
    def __init__(self):
        # 初始化子模块
        self.base_data_retriever = BaseDataRetriever()
        self.comm_manager = CommunicationManager()
        self.annotator = BoundingBoxAnnotator()

    async def pipeline(self):
        base_data_dict = await self.base_data_retriever()

        result = grpc_client.process(base_data_dict)

        if CONFIG['mode'] == 'CO':
            await self.comm_manager.execute_communication(result)

        self.annotator(result)

# ================= 主事件循环 =================

from ROSWrapper import ROSWrapper

async def main_loop():
    await ICP_handler.setup()

    ros_wrapper = ROSWrapper()

    setup_logging()  # 再初始化一次 ROS会莫名其妙的顶掉这个东西

    collaboration_system = CollaborationSystem()

    # 定时处理任务
    async def interval_processing():
        while True:
            start = asyncio.get_running_loop().time()

            await agent_server.clean_timeout_conn()  # 必须先清理完超时连接
            asyncio.create_task(collaboration_system.pipeline())  # 多协程并行

            elapsed = asyncio.get_running_loop().time() - start
            await asyncio.sleep(CONFIG['processing']['interval'] - elapsed)

    # 创建后台任务
    tasks = [
        asyncio.create_task(ICP_handler.recv_loop()),
        asyncio.create_task(agent_server.recv_loop()),
        asyncio.create_task(ros_wrapper.run_in_thread()),
        asyncio.create_task(interval_processing()),
    ]

    await asyncio.gather(*tasks)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="程序配置设置")

    parser.add_argument('--id', type=int, default=CONFIG['id'],
                      help=f"自车 id，默认：{CONFIG['id']}")
    
    parser.add_argument('--mode', type=str, default=CONFIG['mode'],
                      help=f"自车 id，默认：{CONFIG['mode']}")


    parser.add_argument('--zmq_in_port', type=int, default=CONFIG['zmq']['in_port'],
                      help=f"ZMQ 输入端口，默认：{CONFIG['zmq']['in_port']}")
    parser.add_argument('--zmq_out_port', type=int, default=CONFIG['zmq']['out_port'],
                      help=f"ZMQ 输出端口，默认：{CONFIG['zmq']['out_port']}")
    parser.add_argument('--zmq_max_cache', type=int, default=CONFIG['zmq']['max_cache'],
                      help=f"ZMQ 最大缓存数量，默认：{CONFIG['zmq']['max_cache']}")
    
    parser.add_argument('--ros_pointcloud_topic', type=str, default=CONFIG['ros']['pointcloud_topic'],
                      help=f"ROS点云话题名称，默认：{CONFIG['ros']['pointcloud_topic']}")
    parser.add_argument('--ros_queue_size', type=int, default=CONFIG['ros']['queue_size'],
                      help=f"ROS队列大小，默认：{CONFIG['ros']['queue_size']}")
    
    parser.add_argument('--processing_interval', type=float, default=CONFIG['processing']['interval'],
                      help=f"处理间隔时间（秒），默认：{CONFIG['processing']['interval']}")
    parser.add_argument('--processing_max_workers', type=int, default=CONFIG['processing']['max_workers'],
                      help=f"最大工作线程数，默认：{CONFIG['processing']['max_workers']}")
    
    return parser.parse_args()

if __name__ == "__main__":
    setup_logging()  # 在程序入口调用日志配置
    args = parse_args()

    # 更新命令行参数
    if args.zmq_in_port: CONFIG['zmq']['in_port'] = args.zmq_in_port
    if args.zmq_out_port: CONFIG['zmq']['out_port'] = args.zmq_out_port
    if args.zmq_max_cache: CONFIG['zmq']['max_cache'] = args.zmq_max_cache
    if args.ros_pointcloud_topic: CONFIG['ros']['pointcloud_topic'] = args.ros_pointcloud_topic
    if args.ros_queue_size: CONFIG['ros']['queue_size'] = args.ros_queue_size
    if args.processing_interval: CONFIG['processing']['interval'] = args.processing_interval
    if args.processing_max_workers: CONFIG['processing']['max_workers'] = args.processing_max_workers

    # 配置更新日志记录
    config_logger = logging.getLogger('Config')
    config_logger.info("当前配置：%s", json.dumps(CONFIG, indent=2))

    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logging.info("用户主动终止程序")
    finally:
        logging.info("==== 系统关闭 ====")