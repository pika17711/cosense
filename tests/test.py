import subprocess
import time

num = 1
rdps = []
agents = []
agent_files = []  # 存储文件对象，用于后续关闭

def create_agent(i, zmq_in_port, zmq_out_port):
    topic = f'pcd_topic{i}'
    # 启动 RDP 进程
    rdp = subprocess.Popen(['python3', 'test/ros_debug_publisher.py', '--topic_name', topic])
    rdps.append(rdp)
    
    # 为每个 agent 创建独立的日志文件
    stdout_file = open(f'test/agent_{i}_stdout.log', 'w')  # 标准输出日志
    stderr_file = open(f'test/agent_{i}_stderr.log', 'w')  # 标准错误日志
    
    # 启动 agent 并重定向输出
    agent = subprocess.Popen(
        ['python3', 'co/main.py', '--ros_pointcloud_topic', topic, '--mode', 'CO', "--zmq_in_port", 
         str(zmq_in_port), "--zmq_out_port", str(zmq_out_port)],
        stdout=stdout_file,
        stderr=stderr_file
    )
    agents.append(agent)
    agent_files.append((stdout_file, stderr_file))  # 保存文件对象

try:
    create_agent(0, 13415, 13416)
    create_agent(1, 13416, 13415)

    # 主循环保持运行
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    # 终止所有子进程并关闭文件
    for i in range(num):
        rdps[i].kill()
        agents[i].kill()
        agent_files[i][0].close()  # 关闭标准输出文件
        agent_files[i][1].close()  # 关闭标准错误文件
    print("所有进程已终止，日志文件已保存。")