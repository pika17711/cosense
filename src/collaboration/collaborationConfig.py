from dataclasses import dataclass

@dataclass
class CollaborationConfig:
    id = '京A1234'
    app_ver = 1
    app_id = 131
    topic = "W"

    ros_pcd_topic = 'rslidar_points'
    ros_queue_size = 10

    data_cache_size = 1000

    bcctx_keepalive = 120 * 1000   # ms
    cctx_keepalive = 60 * 1000    # ms

    broadcastpub_period = 10000  # ms
    broadcastsub_period = 10000  # ms
    send_data_period = 10000     # ms

    tx_timeout = 50              # ms

    debug = False

    message_max_workers = 100
