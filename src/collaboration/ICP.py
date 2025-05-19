from appConfig import AppConfig
import InteroperationApp.czlconfig
from InteroperationApp.module.zmq_server import ICPServer, ICPClient


def ICP_init(cfg: AppConfig):
    InteroperationApp.czlconfig.source_id = cfg.id

    icp_server = ICPServer(cfg.app_id)
    icp_client = ICPClient(cfg.topic)

    return icp_client, icp_server
