from dataclasses import dataclass, fields

@dataclass
class AppConfig:
    id = '京A1234'
    app_ver = 1
    app_id = 131
    mode = "CO"
    topic = "W"


    ros_pcd_topic = 'rslidar_points'
    ros_queue_size = 10

    message_max_workers = 20

    data_cache_size = 1000

    bcctx_keepalive = 120 * 1000   # ms
    cctx_keepalive = 60 * 1000    # ms

    broadcastpub_period = 10 * 1000  # ms
    broadcastsub_period = 10 * 1000  # ms
    send_data_period = 10 * 1000     # ms

    tx_timeout = 50              # ms

    close_timeout = 0.5           # s

    collaboration_debug = True
    rpc_collaboration_server_debug = False
    rpc_collaboration_client_debug = False

    rpc_perception_server_debug = False
    rpc_perception_client_debug = False

    other_data_cache_ttl = 20    # s

    overlap_threshold = 0.3

    model_dir = 'opencood/logs/point_pillar_where2comm_2024_10_28_23_24_50/'