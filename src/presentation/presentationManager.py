import time
import cv2

from appConfig import AppConfig
from presentation.presentationRPCServer import PresentationRPCServerThread
from presentation.presentationFlaskServer import PresentationFlaskServerThread
from detection.detectionRPCClient import DetectionRPCClient
from utils.sharedInfo import SharedInfo


class PresentationManager:
    def __init__(self, opt, cfg: AppConfig):
        self.shared_info = SharedInfo()
        self.running = False
        self.opt = opt
        self.cfg = cfg
        self.__grpc_prepare()
        self.presentation_flask_server = PresentationFlaskServerThread(self.cfg, self.shared_info)

    def __grpc_prepare(self):
        self.presentation_rpc_server = PresentationRPCServerThread()

        self.detection_client = DetectionRPCClient()

    def start(self):
        self.running = True
        self.presentation_rpc_server.start()
        self.presentation_flask_server.start()
        self.__loop()

    def __loop(self):
        loop_time = 0.33
        last_t = time.time() - loop_time
        while self.running:
            t = time.time()
            if t - last_t < loop_time:
                time.sleep(loop_time + last_t - t)
            t = time.time()
            # print(f'last loop time: {t - last_t}s')
            last_t = t
            presentation_info = self.detection_client.get_presentation_info()
            if presentation_info is not None:
                self.shared_info.update_presentation_info_dict(presentation_info)

            # pcd_img = self.shared_info.get_presentation_info_copy().get('pcd_img')
            # if pcd_img is not None:
            #     img_cv = cv2.cvtColor(pcd_img, cv2.COLOR_BGR2RGB)
            #     cv2.imshow('pcd', img_cv)
            #     cv2.waitKey()

    def close(self):
        self.running = False
        self.presentation_rpc_server.close()
