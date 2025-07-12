import sys
import os
import logging

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from appConfig import AppConfig
import argparse
from detection.detectionManager import DetectionManager


def main():
    cfg = AppConfig()

    opt = argparse.Namespace()
    opt.fusion_method = 'intermediate'
    opt.model_dir = cfg.model_dir
    opt.show_vis = True

    detection_manager = DetectionManager(opt, cfg)
    try:
        detection_manager.start()
    except KeyboardInterrupt:
        print('接收到 Ctrl + C，程序退出中...')
        detection_manager.close()
        logging.info("接收到 Ctrl + C，程序退出。")


if __name__ == '__main__':
    main()
