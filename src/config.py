from dataclasses import dataclass
import importlib
import importlib.util

@dataclass
class AppConfig:
    id = 12345678
    app_ver = 1
    app_id = 131
    mode = "CO"
    topic = ""
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

    broadcastpub_period = 1000 # ms
    broadcastsub_period = 1000 # ms
    send_data_period = 1000 # ms

    tx_timeout = 5000 # ms


def parse_config_file(file_path):
    spec = importlib.util.spec_from_file_location("config", file_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    for attr_name in dir(config_module):
        if not attr_name.startswith("__"):
            attr_value = getattr(config_module, attr_name)
            if hasattr(AppConfig, attr_name):
                setattr(AppConfig, attr_name, attr_value)