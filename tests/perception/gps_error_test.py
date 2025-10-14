import json
import time

import numpy as np
import matplotlib.pyplot as plt
from opencood.utils.transformation_utils import gps_to_utm_transformation, gps_to_enu_transformation

if __name__ == '__main__':
    gps_location = []

    with open('./gps_output_2.txt', 'r') as file:
        for line in file:
            line = json.loads(line)
            if len(line) == 6:
                gps_location.append(line)

    gps_location = np.array(gps_location)[:, :2]

    # 生成示例数据（实际使用时替换为你的真实数据）
    N = 5000  # 数据点数量
    gps_location = np.random.randn(N, 6) * 1  # 生成随机数据
    print(gps_location)

    # 提取水平位置数据 (x, y)
    xy_data = gps_location[:, :2]  # 形状为 [N, 2]

    # 以第一个点为原点进行坐标转换
    origin = xy_data[0]  # 原点坐标
    relative_xy = xy_data - origin  # 所有点相对于原点的坐标

    # 创建图形
    plt.figure(figsize=(16, 9))

    # 绘制原点（黑色圆点）
    plt.scatter(relative_xy[0, 0], relative_xy[0, 1],
                s=100, c='black', marker='o',
                label='Origin (First Point)', zorder=3)

    # 绘制其余点（红色圆点）
    plt.scatter(relative_xy[1:, 0], relative_xy[1:, 1],
                s=40, c='red', marker='o',
                alpha=0.7, label='GPS Points')

    # 添加坐标轴标签和标题
    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Y Position (m)', fontsize=12)
    plt.title('GPS Positioning Error Relative to First Measurement', fontsize=14)

    # 设置等比例坐标系（重要！保证距离比例正确）
    plt.axis('equal')

    # 添加图例和网格
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)

    # # 添加参考距离标注
    # ref_distance = 10  # 参考距离10米
    # plt.annotate(f'Reference Distance: {ref_distance} m',
    #              xy=(0, 0), xytext=(ref_distance, 0),
    #              arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5),
    #              fontsize=10, ha='center')

    # 显示图形
    plt.tight_layout()
    plt.show()
