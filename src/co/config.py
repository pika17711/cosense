
CONFIG = {
    "id": 12345678,
    "mode": "CO",
    "keepalive_timeout": 1,
    "zmq": {
        "in_port": 5555,
        "out_port": 5556,
        "max_cache": 1000
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