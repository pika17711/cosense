from ICP.ICP import ICPServer
from ICP import config

config.send_sub_port = 55566
config.recv_pub_port = 55455
appid = 131
icp_server = ICPServer(appid)
oid = 'ooo'
topic = "where2comm"
icp_server.brocastPub(1, oid, topic, '', 1)

