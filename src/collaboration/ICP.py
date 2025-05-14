from config import AppConfig
from collaboration.collaborationConfig import CollaborationConfig
import InteroperationApp.config
from InteroperationApp.module.zmq_server import ICPServer, ICPClient

icp_server = None
icp_client = None

def ICP_init(cfg: CollaborationConfig):
    InteroperationApp.config.source_id = cfg.id
    global icp_server
    global icp_client

    icp_server = ICPServer(cfg.app_id)
    icp_client = ICPClient(cfg.topic)
