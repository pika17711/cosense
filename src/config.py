
CONFIG = {
    "id": 12345678,
    "app_ver": 1,
    "app_id": 131,
    "mode": "CO",
    "keepalive_timeout": 1,
    "zmq": {
        "in_port": 5555,
        "in_ip": 'localhost',
        "out_port": 5556,
    },
    "ros": {
        "pointcloud_topic": "rslidar_points",
        "queue_size": 10
    },
    "processing": {
        "interval": 1,
        "max_workers": 4,
        "overlap_threshold": 0.3,
        "H": 100,
        "W": 100,
    }
}

from dataclasses import dataclass


@dataclass
class AppConfig:
    id = 12345678
    app_ver = 1
    app_id = 131
    mode = "CO"
    topic = "where2comm"
    zmq_in_port = 5555
    zmq_in_host = '127.0.0.1'
    zmq_out_port = 5556
    zmq_out_host = '127.0.0.1'

    ros_pcd_topic = 'rslidar_points'
    ros_queue_size = 10

    data_cache_size = 1000

    tx_keepalive = 5 * 1000      # transaction keepalive time ms
    bcctx_keepalive = 5 * 1000 # ms
    cctx_keepalive = 2 * 1000    # ms
    csctx_keepalive = 1000       # ms

    id_t = str
    cid_t = int
    sid_t = int