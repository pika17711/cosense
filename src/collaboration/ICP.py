from config import AppConfig
from collaboration.collaborationConfig import CollaborationConfig
import InteroperationApp.config
from InteroperationApp.module.zmq_server import ICPServer, ICPClient


def ICP_init(cfg: CollaborationConfig):
    InteroperationApp.config.source_id = cfg.id

    icp_server = ICPServer(cfg.app_id)
    icp_client = ICPClient(cfg.topic)

    return icp_client, icp_server
