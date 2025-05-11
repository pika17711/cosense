import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import argparse
from detection.detectionManager import DetectionManager

if __name__ == '__main__':
    opt = argparse.Namespace()
    opt.fusion_method = 'intermediate'
    opt.model_dir = 'opencood/logs/point_pillar_where2comm_2024_10_28_23_24_50/'
    opt.show_vis = False

    detectionManager = DetectionManager(opt)
    detectionManager.loop()
