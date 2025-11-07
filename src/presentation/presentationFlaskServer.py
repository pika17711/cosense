import threading

from presentation.flask_app.app import create_app
from collaboration.collaborationRPCClient import CollaborationRPCClient
from appConfig import AppConfig
from utils.sharedInfo import SharedInfo
from queue import Queue


class PresentationFlaskServerThread:                           # 信息呈现子系统的FlaskServer线程
    def __init__(self, cfg: AppConfig, shared_info: SharedInfo, log_queue: Queue, collaboration_rpc_client: CollaborationRPCClient):
        self.flask_app = create_app(cfg, shared_info, log_queue, collaboration_rpc_client)
        self.stop_event = threading.Event()
        self.run_thread = threading.Thread(target=self.run, name='presentation flask server', daemon=True)

    def run(self):
        self.flask_app.run(host='0.0.0.0', debug=False, threaded=True)

    def start(self):
        self.run_thread.start()

    def close(self):
        self.stop_event.set()  # 设置停止标志
