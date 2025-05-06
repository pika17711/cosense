from rpc import Service_pb2
from rpc import Service_pb2_grpc
import threading
import grpc
from concurrent import futures


class PresentationService(Service_pb2_grpc.PresentationServiceServicer):  # 信息呈现子系统的Service类
    pass


class PresentationServerThread(threading.Thread):  # 信息呈现子系统的服务器线程
    def run(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
            ('grpc.max_send_message_length', 64 * 1024 * 1024),                     # 设置gRPC 消息的最大发送和接收大小为64MB
            ('grpc.max_receive_message_length', 64 * 1024 * 1024)])
        Service_pb2_grpc.add_PresentationServiceServicer_to_server(PresentationService(), server)
        server.add_insecure_port('[::]:50054')
        server.start()  # 非阻塞, 会实例化一个新线程来处理请求
        print("Presentation Server is up and running on port 50054.")
        try:
            server.wait_for_termination()  # 保持服务器运行直到终止
        except KeyboardInterrupt:
            server.stop(0)  # 服务器终止
            print("Presentation Server terminated.")


class PresentationClient:  # 信息呈现子系统的Client类，用于请求其他进程的服务
    pass
