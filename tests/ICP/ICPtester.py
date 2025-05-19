import logging
import time
from InteroperationApp.module.zmq_server import ICPServer, ICPClient
import numpy as np

appid = 131
oid = 'ooo'
topic = "W"
icp_client = ICPClient(topic)
icp_server = ICPServer(appid)
logging.basicConfig(level=logging.DEBUG)
data = np.array([1, 2, 3]).tobytes()
icp_server.brocastSub(1, "京A1234", topic, "a" * 32, data, 1, 1)
mes = icp_client.recv_message()
print(mes)