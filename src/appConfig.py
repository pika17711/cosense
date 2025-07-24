from dataclasses import dataclass, fields

@dataclass
class AppConfig:
    """
        配置类
            内存中存储配置的类
            TODO：增加配置文件（json、yaml甚至txt），增加配置文件处理模块
    """
    id = '京A1234'       # 自身id 目前使用车牌号
    app_ver = 1         # 应用版本
    app_id = 131        # 应用id，目前用于标识使用哪个算法
    topic = "W"         # 主题，用于筛选接收的消息，具体机制与zmq SUB PUB中的topic一致

    ros_pcd_topic = 'rslidar_points'  # ROS发布点云的topic名
    ros_queue_size = 10               # 接收ROS收到的点云的队列大小，0表示无限

    message_max_workers = 20          # 消息处理线程池大小，目前消息处理是同步的，未使用

    other_data_cache_size = 1000      # 他车缓存数据个数，超出后按照获取时间淘汰
    other_data_cache_ttl = 20         # 他车缓存数据存活时间，单位是s

    bcctx_keepalive = 120 * 1000      # 广播会话的存活时间，单位ms
    cctx_keepalive = 60 * 1000        # 协作会话的存活时间，单位ms

    broadcastpub_period = 3 * 1000   # 广播推送的发送间隔，单位ms
    broadcastsub_period = 3 * 1000   # 广播订阅的发送间隔，单位ms
    send_data_period = 1 * 1000      # 想订阅者发送数据的间隔，单位ms

    tx_timeout = 50                   # 事务超时时间，单位ms，目前事务是同步的，
                                      # 所以这个超时时间=调用tx_handler事务方法的阻塞时间，所以不要设太久

    close_timeout = 0.5               # 关闭服务器，各个线程和线程池的关闭等待时间，单位是s

    collaboration_debug = False        # collaboration子系统是否开启debug模式，
                                      # 这个配置以及下面的两个配置均是为debug而生的，使有些地方的线程池调用变为同步调用，
                                      # 使有些地方的RPC server使用固定的数据（下方static_asset_path）
                                      # 使有些地方的RPC client不使用RPC，而使用固定的数据（下方static_asset_path）
                                      # 方便debug，之后即可删除
    rpc_collaboration_server_debug = False  # collaboration子系统RPC server是否开启debug模式
    rpc_collaboration_client_debug = False  # collaboration子系统RPC client是否开启debug模式

    collaboration_request_map_debug = True
    collaboration_pcd_debug = True

    perception_debug = False          # 同上
    perception_hea_debug = True
    rpc_perception_server_debug = False
    rpc_perception_client_debug = False

    overlap_threshold = 0.0           # 协作图的重叠率阈值，高于此阈值才建立会话

    # model_dir = 'opencood/logs/point_pillar_where2comm_2024_10_28_23_24_50/'  # 模型位置
    model_dir = r'D:\WorkSpace\Python\interopera\opencood\logs\point_pillar_where2comm_2024_10_28_23_24_50'
    obu_output_file_path = '/home/nvidia/mydisk/czl/InteroperationApp/data/output.json'       # OBU获取的数据导出的文件位置
    # static_asset_path = 'datasets/OPV2V/test_culver_city_part/2021_09_03_09_32_17' + '/302'  # 静态数据位置
    static_asset_path = 'D:\\Documents\\datasets\\OPV2V\\test_tmp\\two\\2021_09_03_09_32_17\\' + '302'
    # static_asset_path = 'D:\\WorkSpace\\Python\\cosense\\tests\\pcds\\25_07_09\\199\\json\\301.json'

    perception_debug_data_from_OPV2V = 'OPV2V' in static_asset_path
