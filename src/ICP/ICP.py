from config import AppConfig
from comm.module.zmq_server import ICPServer, ICPClient

icp_server = ICPServer(AppConfig.app_id)
icp_client = ICPClient(AppConfig.topic)