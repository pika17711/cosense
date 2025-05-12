from dataclasses import dataclass, fields

@dataclass
class AppConfig:
    id = '京A1234'
    app_ver = 1
    app_id = 131
    mode = "CO"
    topic = "W"
    zmq_in_port = 5555
    zmq_in_host = '127.0.0.1'
    zmq_out_port = 5556
    zmq_out_host = '127.0.0.1'

    ros_pcd_topic = 'rslidar_points'
    ros_queue_size = 10

    data_cache_size = 1000

    bcctx_keepalive = 120 * 1000   # ms
    cctx_keepalive = 60 * 1000    # ms

    id_t = str
    cid_t = str
    sid_t = str
    timestamp_t = int

    broadcastpub_period = 10000  # ms
    broadcastsub_period = 10000  # ms
    send_data_period = 10000     # ms

    tx_timeout = 50              # ms

    debug = False