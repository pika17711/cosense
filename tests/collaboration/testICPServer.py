
import base64
import json
import logging
import os
import sys
import time
from typing import List
import zmq
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
from src.collaboration.messageID import MessageID

class TestICPServer:
    def __init__(self, app_id: int, oid: str):
        """
        初始化 ICPServer 类，绑定到指定端口。
        :param port: 服务器端口号 (此参数在原始代码中未使用，已移除)
        :param app_id: 应用标识符
        """
        if app_id is None:
            logging.error("app_id 不能为空！请提供一个有效的应用标识符。")
            self.app_id = -1 # 表示无效状态
        else:
            self.app_id = app_id
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)

        self.socket.connect("tcp://127.0.0.1:5555")
        time.sleep(1)
        self.register_with_broker(app_id, oid)  # 新增注册方法
        
    def register_with_broker(self, app_id: int, oid: str):
        """向Broker注册应用"""
        register_msg = {
            "type": "register",
            "app_id": app_id,
            "oid": oid
        }
        self.send(register_msg)  # 使用原有send方法发送注册消息

    def send(self, message: dict):
        logging.info(f"Sending message: {json.dumps(message, ensure_ascii=False)}") 
        try:
            self.socket.send_string(json.dumps(message, ensure_ascii=False))
            #status = self.socket.getsockopt(zmq.EVENTS)
            #logging.info(f"Socket status: {status}")
        except zmq.error.ZMQError as e:
            logging.error(f"Failed to send message via ZMQ: {e}. Message: {json.dumps(message, ensure_ascii=False)}")
        except TypeError as e: # If message is not JSON serializable
            logging.error(f"Failed to serialize message to JSON: {e}. Message structure: {message}")
        except Exception as e:
            logging.error(f"An unexpected error occurred during send: {e}")


    def AppMessage(self, 
                   CapID:int,
                   CapVersion:int,
                   CapConfig:int,
                   act:int,
                   tid:int =0
                   ):
        """
        构建应用消息
        :param CapID: 能力ID
        :param CapVersion: 能力版本
        :param act: 操作
        """
        if CapID is None or CapVersion is None or act is None or CapConfig is None or tid is None:
            logging.error("AppMessage: CapID, CapVersion, CapConfig, Action 和 tid 不能为空！请提供有效的数据。")
            return
        CapID = CapID & 0xFFFF
        CapVersion = CapVersion & 0xF
        CapConfig = CapConfig & 0xFFF
        message = {
            "mid":MessageID.APPREG.value,
            "app_id": self.app_id,
            "tid": tid,
            "msg":{
                "capId": CapID,
                "capVersion": CapVersion,
                "capConfig": CapConfig,
                "act": act
            }
        }
        self.send(message)
    
    def brocastPub(self,
                   tid:int =0, 
                   oid:str = "",
                   topic:int = 0,
                   coopMap:bytes = b'',
                   coopMapType:int = 0
                   ):
        """
        广播发布消息
        :param tid: 事物标识（transaction id）	应用与控制层之间的请求与应答消息的关联，随机初始化并自增，保证唯一
        :param oid: 源端标识	广播SUB的源端节点标识（仅通信控制->应用接口携带）
        :param topic: 能力标识	广播订购的topic
        :param coopMap: 置信图/协作图 (binary data) 携带用于发送的置信图或协作图
        """
        if oid is None or topic is None or coopMap is None or coopMapType is None: # tid 有默认值，一般不会是None
            logging.error("brocastPub: oid, topic, coopMap 和 coopMapType 不能为空！请提供有效的数据。")
            return
        
        coopMap_str = ""
        try:
            coopMap_str = base64.b64encode(coopMap).decode('utf-8')
        except TypeError:
            logging.error(f"brocastPub: coopMap must be bytes-like for Base64 encoding, got {type(coopMap)}.")
            return
        except Exception as e:
            logging.error(f"brocastPub: Error during Base64 encoding of coopMap: {e}")
            return
        
        message = {
            "mid":MessageID.BROCASTPUB.value,
            "app_id": self.app_id,
            "tid": tid,
            "msg":{
                "oid": oid,
                "topic": topic,
                "coopMap": coopMap_str,
                "coopMapType": coopMapType
            }
        }
        self.send(message)
    
    def brocastSub(self,
                   tid:int =0,
                   oid:str = "",
                   topic:int = 0,
                   context:str = "",
                   coopMap:bytes = b'',
                   coopMapType:int = 0,
                   bearCap:int = 0
                   ):
        """
        广播订购消息
        :param tid: 事物标识（transaction id）	应用与控制层之间的请求与应答消息的关联，随机初始化并自增，保证唯一
        :param oid: 源端标识	广播SUB的源端节点标识（仅通信控制->应用接口携带）
        :param topic: 能力标识	广播订购的topic
        :param context: 会上下文标识	应用创建的用于区分对话的标识
        :param coopMap: 置信图/协作图 (binary data) 携带用于发送的置信图或协作图
        :param bearcap: 承载能力描述	1：要求携带用于描述自身承载能力的信息
        """
        if oid is None or topic is None or context is None or coopMap is None or bearCap is None or coopMapType is None:
            logging.error("brocastSub: oid, topic, context, coopMap, coopMapType 和 bearCap 不能为空！请提供有效的数据。") 
            return

        coopMap_str = ""
        try:
            coopMap_str = base64.b64encode(coopMap).decode('utf-8')
        except TypeError:
            logging.error(f"brocastSub: coopMap must be bytes-like for Base64 encoding, got {type(coopMap)}.")
            return
        except Exception as e:
            logging.error(f"brocastSub: Error during Base64 encoding of coopMap: {e}")
            return

        message = {
            "mid":MessageID.BROCASTSUB.value,
            "app_id": self.app_id,
            "tid": tid,
            "msg":{
                "oid": oid,
                "topic": topic,
                "context": context,
                "coopMap": coopMap_str,
                "coopMapType": coopMapType,
                "bearCap": bearCap
            }
        }
        self.send(message)
    
    def brocastSubnty(self,
                     tid:int =0,
                     oid:str = "",
                     did:str = "",
                     topic:int = 0,
                     context:str = "",
                     coopMap:bytes = b'', 
                     coopMapType:int = 0,
                     bearCap:int = 0
                     ):
        """
        广播订购通知消息 (brocastSubNotify seems to be the intended name based on config.boardCastSubNotify)
        :param tid: 事物标识（transaction id）	应用与控制层之间的请求与应答消息的关联，随机初始化并自增，保证唯一
        :param oid: 源端标识	广播SUB的源端节点标识（仅通信控制->应用接口携带）
        :param did: 目的端标识	广播SUB的目的端节点标识（仅通信控制->应用接口携带）
        :param topic: 能力标识	广播订购的topic
        :param context: 会上下文标识	应用创建的用于区分对话的标识
        :param coopMap: 置信图/协作图 (binary data) 携带用于发送的置信图或协作图
        :param bearcap: 承载能力描述	1：要求携带用于描述自身承载能力的信息
        """
        if oid is None or did is None or topic is None or context is None or coopMap is None or bearCap is None or coopMapType is None:
            logging.error("brocastSubnty: oid, did, topic, context, coopMap, coopMapType 和 bearCap 不能为空！请提供有效的数据。") 
            return

        coopMap_str = ""
        try:
            coopMap_str = base64.b64encode(coopMap).decode('utf-8')
        except TypeError:
            logging.error(f"brocastSubnty: coopMap must be bytes-like for Base64 encoding, got {type(coopMap)}.")
            return
        except Exception as e:
            logging.error(f"brocastSubnty: Error during Base64 encoding of coopMap: {e}")
            return

        message = {
            "mid":MessageID.BROCASTSUBNTY,
            "app_id": self.app_id,
            "tid": tid,
            "msg":{
                "oid": oid,
                "did": did,
                "topic": topic,
                "context": context,
                "coopMap": coopMap_str, 
                "coopMapType": coopMapType,
                "bearCap": bearCap
            }
        }
        self.send(message)      
        
    def subMessage(self,
                   tid:int =0,
                   oid:str = "",
                   did: List[str] = [],
                   topic:int = 0,
                   act:int = 0,
                   context:str = "",
                   coopMap:bytes = b'',
                   coopMapType:int = 0,
                   bearInfo:int = 0
                   ):
        """
        订购消息
        :param tid: 事物标识（transaction id）	应用与控制层之间的请求与应答消息的关联，随机初始化并自增，保证唯一
        :param oid: 源端标识	广播SUB的源端节点标识（仅通信控制->应用接口携带）
        :param did: 目的端标识	广播SUB的目的端节点标识（仅通信控制->应用接口携带）
        :param topic: 能力标识	广播订购的topic
        :param context: 会上下文标识	应用创建的用于区分对话的标识
        :param coopMap: 置信图/协作图 (binary data) 携带用于发送的置信图或协作图
        :param bearInfo: 承载地址描述	1：要求携带用于描述自身承载地址的信息
        """
        if oid is None or did is None or topic is None or context is None or coopMap is None or bearInfo is None or coopMapType is None or act is None:
            logging.error("subMessage: oid, did, topic, act, context, coopMap, coopMapType 和 bearInfo 不能为空！请提供有效的数据。") 
            return

        coopMap_str = ""
        try:
            coopMap_str = base64.b64encode(coopMap).decode('utf-8')
        except TypeError:
            logging.error(f"subMessage: coopMap must be bytes-like for Base64 encoding, got {type(coopMap)}.")
            return
        except Exception as e:
            logging.error(f"subMessage: Error during Base64 encoding of coopMap: {e}")
            return
        
        message = {
            "mid": MessageID.SUBSCRIBE.value,
            "app_id": self.app_id,
            "tid": tid,
            "msg":{
                "oid": oid,
                "did": did,
                "topic": topic,
                "act": act,
                "context": context,
                "coopMap": coopMap_str, 
                "coopMapType": coopMapType,
                "bearinfo": bearInfo 
            }
        }
        self.send(message)
        
    def notifyMessage(self,
                      tid:int =0,
                      oid:str = "",
                      did:str = "",
                      topic:int = 0,
                      act:int = 0,
                      context:str = "",
                      coopMap:bytes = b'',
                      coopMapType:int = 0,
                      bearCap:int = 0
                      ):
        """
        通知消息
        :param tid: 事物标识（transaction id）	应用与控制层之间的请求与应答消息的关联，随机初始化并自增，保证唯一
        :param oid: 源端标识	广播SUB的源端节点标识（仅通信控制->应用接口携带）
        :param did: 目的端标识	广播SUB的目的端节点标识（仅通信控制->应用接口携带）
        :param topic: 能力标识	广播订购的topic
        :param context: 会上下文标识	应用创建的用于区分对话的标识
        :param coopMap: 置信图/协作图 (binary data) 携带用于发送的置信图或协作图
        :param bearCap: 承载能力描述	1：要求携带用于描述自身承载能力的信息
        """
        if oid is None or did is None or topic is None or context is None or coopMap is None or bearCap is None or coopMapType is None or act is None:
            logging.error("notifyMessage: oid, did, topic, act, context, coopMap, coopMapType 和 bearCap 不能为空！请提供有效的数据。") 
            return

        coopMap_str = ""
        try:
            coopMap_str = base64.b64encode(coopMap).decode('utf-8')
        except TypeError:
            logging.error(f"notifyMessage: coopMap must be bytes-like for Base64 encoding, got {type(coopMap)}.")
            return
        except Exception as e:
            logging.error(f"notifyMessage: Error during Base64 encoding of coopMap: {e}")
            return

        message = {
            "mid": MessageID.NOTIFY.value,
            "app_id": self.app_id,
            "tid": tid,
            "msg":{ 
                "oid": oid,
                "did": did,
                "topic": topic,
                "act": act,
                "context": context,
                "coopMap": coopMap_str,
                "coopMapType": coopMapType,
                "bearCap": bearCap
            }
        }
        self.send(message)
    
    def streamSendreq(self,
                      did:str = "",
                      context:str = "",
                      rl:int =1,
                      pt:int = 0
                      ):
        """
        流发送请求
        :param did: 目的端标识	广播SUB的目的端节点标识（仅通信控制->应用接口携带）
        :param context: 会上下文标识	应用创建的用于区分对话的标识
        :param rl: 流数据的质量保证	雷达数据等默认需要 RL=1
        :param pt: 数据类型	请求数据类型
        """
        if did is None or context is None or rl is None or pt is None: 
            logging.error("streamSendreq: did, context, rl 和 pt 不能为空！请提供有效的数据。") 
            return
        message = {
            "mid": MessageID.SENDREQ.value,
            "app_id": self.app_id,
            "msg":{
                "did": did,
                "context": context,
                "rl": rl,
                "pt": pt
            }
        }
        self.send(message)

    def streamSend(self,
                   sid:str = "",
                   data:str = ""
                   ):
        """
        流发送
        :param sid: 流标识	流标识
        :param data: 数据	发送的数据
        """
        if sid is None or data is None:
            logging.error("streamSend: sid 和 data 不能为空！请提供有效的数据。") 
            return
        message = {
            "mid": MessageID.SEND.value,
            "app_id": self.app_id,
            "msg":{
                "sid": sid,
                "data": data
            }
        }
        self.send(message)

    def streamSendend(self,
                      did:str = "",
                      context:str = "",
                      sid:str = ""
                      ):
        """
        流发送结束
        :param did: 目的端标识	广播SUB的目的端节点标识（仅通信控制->应用接口携带）
        :param context: 会上下文标识	应用创建的用于区分对话的标识
        :param sid: 流标识	流标识
        """
        if sid is None or did is None or context is None:
            logging.error("streamSendend: sid, did 和 context 不能为空！请提供有效的数据。") 
            return
        message = {
            "mid": MessageID.SENDEND.value,
            "app_id": self.app_id,
            "msg":{
                "did": did,
                "context": context,
                "sid": sid
            }
        }
        self.send(message)
    
    def sendFile(self,
                 did:str = "", 
                 context:str = "",
                 rl:int =1, 
                 pt:int = 0,
                 file:str = ""
                 ):
        """
        发送文件
        :param did: 目的端标识	广播SUB的目的端节点标识（仅通信控制->应用接口携带）
        :param context: 会上下文标识	应用创建的用于区分对话的标识
        :param rl: 流数据的质量保证	雷达数据等默认需要 RL=1
        :param pt: 数据类型	请求数据类型
        :param file:文件路径	发送文件的存储路径与文件名
        """
        if did is None or context is None or rl is None or pt is None or file is None:
            logging.error("sendFile: did, context, rl, pt 和 file 不能为空！请提供有效的数据。") 
            return
        message = {
            "mid": MessageID.SENDFILE.value,
            "app_id": self.app_id,
            "msg": {
                "did": did,
                "context": context,
                "rl": rl,
                "pt": pt,
                "file": file
            }
        }
        self.send(message)