import threading
import grpc
import logging
from concurrent import futures

from rpc import Service_pb2
from rpc import Service_pb2_grpc


class PresentationRPCService(Service_pb2_grpc.PresentationServiceServicer):    # 信息呈现子系统的RPCService类
    pass


class PresentationRPCServerThread:                           # 信息呈现子系统的RPCServer线程
    def __init__(self):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
            ('grpc.max_send_message_length', 64 * 1024 * 1024),  # 设置gRPC 消息的最大发送和接收大小为64MB
            ('grpc.max_receive_message_length', 64 * 1024 * 1024)])
        Service_pb2_grpc.add_PresentationServiceServicer_to_server(PresentationRPCService(), self.server)
        self.stop_event = threading.Event()
        self.run_thread = threading.Thread(target=self.run, name='presentation rpc server', daemon=True)

    def run(self):
        self.server.add_insecure_port('[::]:50054')
        self.server.start()  # 非阻塞, 会实例化一个新线程来处理请求
        logging.info("Presentation Server is up and running on port 50054.")
        try:
            # 等待停止事件或被中断
            while not self.stop_event.is_set():
                self.stop_event.wait(1)  # 每1秒检查一次停止标志
        except KeyboardInterrupt:
            pass
        finally:
            # 优雅地关闭服务器
            if self.server:
                self.server.stop(0.5).wait()

    def start(self):
        self.run_thread.start()

    def close(self):
        self.stop_event.set()  # 设置停止标志
