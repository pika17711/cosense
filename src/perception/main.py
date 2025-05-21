import sys
import os
import logging

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from perception.perceptionManager import PerceptionManager


def main():
    perception_manager = PerceptionManager()
    try:
        perception_manager.start()
    except KeyboardInterrupt:
        print('接收到 Ctrl + C，程序退出中...')
        perception_manager.close()
        logging.info("接收到 Ctrl + C，程序退出。")


if __name__ == '__main__':
    main()
