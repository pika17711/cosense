import sys
import os

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
    opt.show_vis = False

    detectionManager = DetectionManager(opt, cfg)
    detectionManager.loop()

if __name__ == '__main__':
    main()