import asyncio
import json
import logging
from typing import Any, Dict, List

import zmq
import zmq.asyncio
from config import CONFIG

import json
import utils

class ICPMessage:
    MTACK = 0 # useless
    MTSUB = 1
    MTPUB = 2
    MTECHO = 3 # for debug

    OPSUBUNDO = 0
    OPSUBDO = 1

    OPPUBPRO = 0
    OPPUBACK = 1
    OPPUBFIN = 2

    def __init__(
        self,
        app_id,
        reliability,
        length,
        message_type,
        data,
        topic,
        qos,
        operator,
        source_id,
        peer_id,
        caps_list=None,
        extension=None
    ):
        self._app_id = app_id
        self._reliability = reliability
        self._length = length
        self._message_type = message_type
        self._data = data
        self._caps_list = caps_list if caps_list is not None else []
        self._topic = topic
        self._qos = qos
        self._operator = operator
        self._source_id = source_id
        self._peer_id = peer_id
        self._extension = extension if extension is not None else {}
        self._timestamp = None

    # Getter and Setter 方法
    @property
    def app_id(self):
        return self._app_id

    @app_id.setter
    def app_id(self, value):
        self._app_id = value

    @property
    def reliability(self):
        return self._reliability

    @reliability.setter
    def reliability(self, value):
        self._reliability = value

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value

    @property
    def message_type(self):
        return self._message_type

    @message_type.setter
    def message_type(self, value):
        self._message_type = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def caps_list(self):
        return self._caps_list

    @caps_list.setter
    def caps_list(self, value):
        self._caps_list = value if value is not None else []

    @property
    def topic(self):
        return self._topic

    @topic.setter
    def topic(self, value):
        self._topic = value

    @property
    def qos(self):
        return self._qos

    @qos.setter
    def qos(self, value):
        self._qos = value

    @property
    def operator(self):
        return self._operator

    @operator.setter
    def operator(self, value):
        self._operator = value

    @property
    def source_id(self):
        return self._source_id

    @source_id.setter
    def source_id(self, value):
        self._source_id = value

    @property
    def peer_id(self):
        return self._peer_id

    @peer_id.setter
    def peer_id(self, value):
        self._peer_id = value

    @property
    def extension(self):
        return self._extension

    @extension.setter
    def extension(self, value):
        self._extension = value if value is not None else {}

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        self._timestamp = value if value is not None else {}

    # 原有 from_json 和 to_json 方法保持不变
    @classmethod
    def from_json(cls, json_str, ts):
        data = json.loads(json_str)
        obj = cls(
            app_id=data["ApplicationIdentifier"],
            reliability=data["Reliability"],
            length=data["Length"],
            message_type=data["Message Type"],
            data=data["Data"],
            topic=data["Topic"],
            qos=data["QoS"],
            operator=data["Operator"],
            source_id=data["Source Vehicle ID"],
            peer_id=data["Peer Vehicle ID"],
            caps_list=data.get("CapsList", []),
            extension=data.get("Extension", {}),
        )
        obj.timestamp = ts
        return obj

    def to_json(self):
        message_dict = {
            "ApplicationIdentifier": self.app_id,
            "Reliability": self.reliability,
            "Length": self.length,
            "Message Type": self.message_type,
            "Data": self.data,
            "CapsList": self.caps_list,
            "Topic": self.topic,
            "QoS": self.qos,
            "Operator": self.operator,
            "Source Vehicle ID": self.source_id,
            "Peer Vehicle ID": self.peer_id,
            "Extension": self.extension
        }
        return json.dumps(message_dict)

    def validate(self) -> bool:
        """验证消息必填字段及取值范围"""
        if not all([self.app_id, self.source_id, self.peer_id]):
            raise ValueError("Missing required fields: app_id/source_id/peer_id")
        
        if self.message_type not in (self.MTSUB, self.MTPUB):
            raise ValueError(f"Invalid message_type: {self.message_type}")
        
        if not 0 <= self.qos <= 2:
            raise ValueError(f"QoS must be 0-2, got {self.qos}")
        
        if self.message_type == self.MTSUB and self.operator not in (self.OPSUBUNDO, self.OPSUBDO):
            raise ValueError(f"Invalid SUB operator: {self.operator}")
            
        if self.message_type == self.MTPUB and self.operator not in (self.OPPUBPRO, self.OPPUBACK, self.OPPUBFIN):
            raise ValueError(f"Invalid PUB operator: {self.operator}")
        
        return True

    @classmethod
    def create_example(cls, message_type: int, operator: int) -> "ICPMessage":
        """快速创建示例消息的工厂方法"""
        return cls(
            app_id="DEMO_APP",
            reliability=90,
            length=256,
            message_type=message_type,
            data={"sample": "data"},
            topic="default_topic",
            qos=1,
            operator=operator,
            source_id="SRC_001",
            peer_id="PEER_002",
            caps_list=["basic_capability"]
        )

    def merge_extension(self, new_extension: Dict[str, Any]) -> None:
        """深度合并扩展字段（保留原有数据）"""
        if not isinstance(new_extension, dict):
            raise TypeError("Extension must be a dictionary")
        self.extension.update(new_extension)

    def set_current_timestamp(self) -> None:
        """设置ISO格式时间戳"""
        self.timestamp = utils.mstime()

    def is_sub(self) -> bool:
        return self.message_type == self.MTSUB

    def is_sub_do(self) -> bool:
        return self.message_type == self.OPSUBDO

    def is_sub_undo(self) -> bool:
        return self.message_type == self.OPSUBUNDO

    def is_pub(self) -> bool:
        return self.message_type == self.MTPUB

    def is_echo(self) -> bool:
        return self.message_type == self.MTECHO

    def create_reply(self, swap_peer: bool = True) -> "ICPMessage":
        """生成回复消息（自动交换源/目标ID）"""
        new_source = self.peer_id if swap_peer else self.source_id
        new_peer = self.source_id if swap_peer else self.peer_id
        
        return ICPMessage(
            app_id=self.app_id,
            reliability=self.reliability,
            length=self.length,
            message_type=self.message_type,
            data=self.data.copy(),
            topic=self.topic,
            qos=self.qos,
            operator=self.OPPUBACK,  # 默认设为确认操作
            source_id=new_source,
            peer_id=new_peer,
            caps_list=self.caps_list.copy(),
            extension=self.extension.copy()
        )

    def to_dict(self, include_private: bool = False) -> Dict[str, Any]:
        """转换为字典（可选包含私有字段）"""
        public_fields = {
            "ApplicationIdentifier": self.app_id,
            "Reliability": self.reliability,
            "Length": self.length,
            "Message Type": self.message_type,
            "Data": self.data,
            "CapsList": self.caps_list,
            "Topic": self.topic,
            "QoS": self.qos,
            "Operator": self.operator,
            "Source Vehicle ID": self.source_id,
            "Peer Vehicle ID": self.peer_id,
            "Extension": self.extension
        }
        if include_private:
            public_fields["_timestamp"] = self.timestamp

        return public_fields

    def filter_fields(self, field_names: List[str]) -> Dict[str, Any]:
        """过滤指定字段生成精简字典"""
        full_dict = self.to_dict()
        return {k: v for k, v in full_dict.items() if k in field_names}

    def update_caps(self, new_caps: List[str], replace: bool = False) -> None:
        """更新能力列表（追加或替换）"""
        if replace:
            self.caps_list = new_caps
        else:
            self.caps_list.extend(new_caps)


class ICPHandler:
    def __init__(self):
        self.ctx = zmq.asyncio.Context()
        self.receiver = None
        self.sender = None
        self.logger = logging.getLogger('ZMQ')

    async def setup(self):

        # 发送端
        self.sender = self.ctx.socket(zmq.PUB)
        self.sender.bind(f"tcp://*:{CONFIG['zmq']['out_port']}")

        # 接收端
        self.receiver = self.ctx.socket(zmq.SUB)
        self.receiver.connect(f"tcp://localhost:{CONFIG['zmq']['in_port']}")
        self.receiver.setsockopt_string(zmq.SUBSCRIBE, '')


        self.msq = asyncio.Queue()  # message queue

    async def recv_loop(self):
        self.logger.info("ZMQ接收循环启动")
        while True:
            try:
                packet = await self.receiver.recv_string()
                msg: ICPMessage = ICPMessage.from_json(packet, utils.mstime())
                self.logger.debug(f"从{msg.source_id} 收到消息长度: {len(packet)} bytes")
                await self.cached_message(msg)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON解析失败: {e.doc}")
            except Exception as e:
                self.logger.exception("ZMQ接收发生异常")

    async def cached_message(self, msg):
        await self.msq.put(msg)

    async def wait_message(self):
        return await self.msq.get()

    async def request(self, msg: ICPMessage):
        try:
            await self.sender.send_string(msg.to_json())
            self.logger.debug(f"向 {msg.peer_id} 发送数据成功")
        except Exception as e:
            self.logger.error(f"数据发送失败: {str(e)}")

    async def pub(self, msg: ICPMessage, operator):
        msg.message_type = ICPMessage.MTPUB
        msg.operator = operator
        await self.request(msg)

    async def pub_pro(self, msg: ICPMessage):
        await self.pub(msg, ICPMessage.OPPUBPRO)

    async def pub_ack(self, msg: ICPMessage):
        await self.pub(msg, ICPMessage.OPPUBACK)

    async def pub_fin(self, msg: ICPMessage):
        await self.pub(msg, ICPMessage.OPPUBFIN)

    async def sub(self, msg: ICPMessage, operator):
        msg.message_type = ICPMessage.MTSUB
        msg.operator = operator
        await self.request(msg)

    async def sub_undo(self, msg: ICPMessage):
        await self.sub(msg, operator=ICPMessage.OPSUBUNDO)

    async def sub_do(self, msg: ICPMessage):
        await self.sub(msg, operator=ICPMessage.OPSUBDO)

    async def echo(self, msg):
        msg.message_type = ICPMessage.MTECHO
        await self.request(msg)

ICP_handler = ICPHandler()