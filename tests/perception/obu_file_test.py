import sys
import os
import time

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.append(parent_dir)

from utils.perception_utils import get_psa_from_obu
from appConfig import AppConfig

if __name__ == '__main__':
    with open('./gps_output_1.txt', 'w') as file:
        while True:
            lidar_pose, _, _ = get_psa_from_obu(AppConfig.obu_output_file_path)
            print(lidar_pose)
            file.writelines(str(lidar_pose.tolist()) + '\n')

            time.sleep(0.1)

