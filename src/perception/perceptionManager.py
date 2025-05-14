import time
from perception.perceptionRPCServer import PerceptionServerThread, MyInfo

def load_pcd(no):
    pcd_load = o3d.io.read_point_cloud('../dataset/0000'+str(no)+'.pcd')

    # 将Open3D的点云对象转换为NumPy数组
    xyz = np.asarray(pcd_load.points)
    colors = np.zeros((len(xyz), 1))
    pcd = np.hstack((xyz, colors))
    return pcd

class PerceptionManager:
    def __init__(self):
        self.my_info = MyInfo()
        self.service1_thread = PerceptionServerThread(self.my_info)
        self.service1_thread.setDaemon(True)
        self.service1_thread.start()

    def loop(self):
        try:
            # 保持主线程存活，防止程序退出
            while True:
                for i in range(4):
                    pcd = load_pcd(68 + 2 * i)
                    self.my_info.update_pcd(pcd)
                    time.sleep(1)
        except KeyboardInterrupt:
            print("Server terminated.")



