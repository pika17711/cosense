import sys
import os
import logging
import argparse

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from appConfig import AppConfig
from presentation.presentationManager import PresentationManager


def main():
    cfg = AppConfig()
    # TODO: load config

    opt = argparse.Namespace()

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("presentation模块启动")

    presentation_manager = PresentationManager(opt, cfg)
    logging.debug("presentation_manager start")
    try:
        presentation_manager.start()
    except KeyboardInterrupt:
        print('接收到 Ctrl + C，程序退出中...')
        presentation_manager.close()
        logging.info("接收到 Ctrl + C，程序退出。")


if __name__ == '__main__':
    main()