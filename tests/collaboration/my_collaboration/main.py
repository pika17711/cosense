import logging
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from CollaborationManager_locally import CollaborationManager
from appConfig import AppConfig


def main():
    cfg = AppConfig()

    collaboration_manager = CollaborationManager(cfg)

    try:
        collaboration_manager.start()
    except KeyboardInterrupt:
        print('接收到 Ctrl + C，程序退出中...')
        collaboration_manager.close()
        logging.info("接收到 Ctrl + C，程序退出。")


if __name__ == '__main__':
    main()
