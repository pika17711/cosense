import threading

from presentation.flask_app.app import create_app
from appConfig import AppConfig
from utils.sharedInfo import SharedInfo


class PresentationFlaskServerThread:                           # 信息呈现子系统的FlaskServer线程
    def __init__(self, cfg: AppConfig, shared_info: SharedInfo):
        self.flask_app = create_app(cfg, shared_info)
        self.stop_event = threading.Event()
        self.run_thread = threading.Thread(target=self.run, name='presentation flask server', daemon=True)

    def run(self):
        self.flask_app.run(debug=False, threaded=True)

    def start(self):
        self.run_thread.start()

    def close(self):
        self.stop_event.set()  # 设置停止标志
