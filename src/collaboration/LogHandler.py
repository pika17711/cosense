import logging
from presentation.presentationRPCClient import PresentationRPCClient


class LogHandler(logging.Handler):
    def __init__(self, presentation_client: PresentationRPCClient):
        super().__init__()
        self.presentation_client = presentation_client

    def emit(self, record):
        log = self.format(record)
        self.presentation_client.send_log(log)


def log_init(presentation_client: PresentationRPCClient):
    """
        日志初始化
    """
    # logging.basicConfig(level=logging.DEBUG,
    #                     filename='collaboration.log',
    #                     filemode='w',
    #                     format='%(asctime)s - %(levelname)s - %(message)s')
    # self.loggerinfo("协同模块启动")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_handler = LogHandler(presentation_client)
    log_handler.setFormatter(formatter)

    logger.addHandler(log_handler)
    return logger


presentation_client = PresentationRPCClient()
logger = log_init(presentation_client)
