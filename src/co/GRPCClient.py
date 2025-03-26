import grpc
import numpy as np
from sense_pb2 import NestedDataRequest
from sense_pb2_grpc import NumpyServiceStub
from serializer import Serializer

class GrpcClient:
    def __init__(self, host='localhost', port=50051):
        self.channel = grpc.insecure_channel(f'{host}:{port}')
        self.stub = NumpyServiceStub(self.channel)
    
    def process(self, input_data: dict) -> dict:
        # 序列化请求
        request = NestedDataRequest(
            items=Serializer.serialize_request(input_data)
        )
        
        # 发送请求
        response = self.stub.ProcessNestedData(request)
        
        # 反序列化响应
        result = {
            item_id: Serializer.deserialize_ndarray(array_proto)
            for item_id, array_proto in response.results.items()
        }

        return result

grpc_client = GrpcClient()