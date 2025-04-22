from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Type, Dict, Any
import json
from datetime import datetime
import zmq
from src.config import CONFIG
from src.mes.mid import MessageID

class MessageError(Exception):
    """消息解析异常基类"""
    pass

class InvalidMessageFormat(MessageError):
    """消息格式错误"""
    pass

class UnknownMessageType(MessageError):
    """未知消息类型"""
    pass

@dataclass
class MessageHeader:
    """消息头基类"""
    mid: MessageID
    appid: int
    tid: Optional[int] = None  # 仅控制消息存在
    
    @classmethod
    def from_dict(cls, data: Dict) -> "MessageHeader":
        try:
            return cls(
                mid=MessageID(data["mid"]),
                appid=data["appid"],
                tid=data.get("tid", None)
            )
        except KeyError as e:
            raise InvalidMessageFormat(f"Missing header field: {e}")

@dataclass
class Message:
    """消息基类"""
    header: MessageHeader
    direction: str  # 消息方向

    @classmethod
    def parse(cls, raw_data: Dict) -> "Message":
        """工厂方法：从原始字典创建具体消息对象"""
        header = MessageHeader.from_dict(raw_data.get("header", {}))
        msg_class = cls._get_message_class(header.mid)
        return msg_class.from_raw(header, raw_data.get("msg", {}))

    @classmethod
    def _get_message_class(cls, mid: MessageID) -> Type["Message"]:
        mapping = {
            MessageID.APPREG: AppRegMessage,
            MessageID.APPRSP: AppRspMessage,
            MessageID.BROCASTPUB: BroadcastPubMessage,
            MessageID.BROCASTSUB: BroadcastSubMessage,
            MessageID.BROCASTSUBNTY: BroadcastSubNtyMessage,
            MessageID.SUBSCRIBE: SubscribeMessage,
            MessageID.NOTIFY: NotifyMessage,
            MessageID.SENDREQ: StreamReqMessage,
            MessageID.SENDRDY: StreamRdyMessage,
            MessageID.RECVRDY: StreamRdyMessage,
            MessageID.SEND: StreamDataMessage,
            MessageID.RECV: StreamDataMessage,
            MessageID.SENDEND: StreamEndMessage,
            MessageID.RECVEND: StreamEndMessage,
            MessageID.SENDFILE: FileReqMessage,
            MessageID.SENDFIN: FileFinMessage,
            MessageID.RECVFILE: FileRecvMessage,
        }
        if msg_class := mapping.get(mid):
            return msg_class
        raise UnknownMessageType(f"Unsupported MID: {mid.name}")


# ----------------------------
# 具体消息类型实现
# ----------------------------

# ----------------------------
# 控制接口消息实现
# ----------------------------
@dataclass
class AckMessage(Message):
    """消息相应 (MID.ACK)"""
    code: int
    mes: str

    @classmethod
    def from_raw(cls, header: MessageHeader, msg_body: Dict) -> "AckMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            code=msg_body['code'],
            mes=msg_body['mes']
        )


@dataclass
class AppRegMessage(Message):
    """应用注册 (MID.APPREG)"""
    topic: int
    ver: int
    act: int
    
    @classmethod
    def from_raw(cls, header: MessageHeader, msg_body: Dict) -> "AppRegMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            topic=msg_body["topic"],
            ver=msg_body["ver"],
            act=msg_body["act"]
        )

@dataclass
class AppRspMessage(Message):
    """注册响应 (MID.APPRSP)"""
    result: int
    
    @classmethod
    def from_raw(cls, header: MessageHeader, msg_body: Dict) -> "AppRspMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            result=msg_body["result"],
        )

@dataclass
class BroadcastPubMessage(Message):
    """广播推送 (MID.BROCASTPUB)"""
    oid: str
    topic: int
    coopmap: bytes
    extra: Optional[dict] = None
    
    @classmethod
    def from_raw(cls, header: MessageHeader, msg_body: Dict) -> "BroadcastPubMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            origin_id=msg_body["oid"],
            topic=msg_body["topic"],
            coopmap=bytes.fromhex(msg_body["coopmap"]),
            extra=msg_body.get("extra")
        )

@dataclass
class BroadcastSubMessage(Message):
    """广播订购 (MID.BROCASTSUB)"""
    oid: int
    topic: int
    context: int
    coopmap: bytes
    bearcap: int
    
    @classmethod
    def from_raw(cls, header: MessageHeader, msg_body: Dict) -> "BroadcastSubMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            oid=msg_body["oid"],
            topic=msg_body["topic"],
            context=msg_body["context"],
            coopmap=bytes.fromhex(msg_body["coopmap"]),
            bearcap=msg_body["bearcap"]
        )

@dataclass
class BroadcastSubNtyMessage(Message):
    """广播订购通知 (MID.BROCASTSUBNTY)"""
    oid: int
    topic: int
    context: int
    coopmap: bytes
    bearcap: int
    
    @classmethod
    def from_raw(cls, header: MessageHeader, msg_body: Dict) -> "BroadcastSubNtyMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            oid=msg_body["oid"],
            topic=msg_body["topic"],
            context=msg_body["context"],
            coopmap=msg_body["coopmap"],
            bearcap=msg_body.get("bearcap")
        )

@dataclass
class SubscribeMessage(Message):
    """能力订购 (MID.SUBSCRIBE)"""
    oid: int
    # did 无
    topic: int
    action: int
    context: int
    coopmap: bytes
    bearinfo: bool
    
    @classmethod
    def from_raw(cls, header: MessageHeader, msg_body: Dict) -> "SubscribeMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            oid=msg_body["oid"],
            topic=msg_body["topic"],
            action=msg_body["act"],
            context=msg_body["context"],
            coopmap=bytes.fromhex(msg_body["coopmap"]),
            bearinfo=bool(msg_body.get("bearinfo", 0))
        )

@dataclass
class NotifyMessage(Message):
    """订购通知 (MID.NOTIFY)"""
    oid: int
    # did 无
    topic: int
    action: int
    context: int
    coopmap: bytes
    bearcap: int
    
    @classmethod
    def from_raw(cls, header: MessageHeader, msg_body: Dict) -> "NotifyMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            oid=msg_body["oid"],
            topic=msg_body["topic"],
            action=msg_body["act"],
            context=msg_body["context"],
            coopmap=bytes.fromhex(msg_body["coopmap"]),
            bearcap=msg_body.get("bearcap")
        )

# ----------------------------
# 流传输消息实现
# ----------------------------
@dataclass
class StreamReqMessage(Message):
    """流发送请求 (MID.SENDREQ)"""
    destination_id: int
    context: int
    qos: int
    data_type: int
    
    @classmethod
    def from_raw(cls, header: MessageHeader, msg_body: Dict) -> "StreamReqMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            destination_id=msg_body["did"],
            context=msg_body["context"],
            qos=msg_body["rl"],
            data_type=msg_body["pt"]
        )

@dataclass
class StreamRdyMessage(Message):
    """流准备就绪 (MID.SENDRDY/MID.RECVRDY)"""
    stream_id: int
    context: int
    partner_id: int
    
    @classmethod
    def from_raw(cls, header: MessageHeader, msg_body: Dict) -> "StreamRdyMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            stream_id=msg_body["sid"],
            context=msg_body["context"],
            partner_id=msg_body["did"] if header.mid == MessageID.SENDRDY else msg_body["oid"]
        )

@dataclass
class StreamDataMessage(Message):
    """流数据消息基类"""
    stream_id: int
    data: bytes
    
    @classmethod
    def from_raw(cls, header: MessageHeader, msg_body: Dict) -> "StreamDataMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            stream_id=msg_body["sid"],
            data=bytes.fromhex(msg_body["data"])
        )

@dataclass
class StreamEndMessage(Message):
    """流结束消息 (MID.SENDEND/MID.RECVEND)"""
    stream_id: int
    context: int
    
    @classmethod
    def from_raw(cls, header: MessageHeader, msg_body: Dict) -> "StreamEndMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            stream_id=msg_body["sid"],
            context=msg_body["context"]
        )

# ----------------------------
# 文件传输消息实现
# ----------------------------
@dataclass
class FileReqMessage(Message):
    """文件发送请求 (MID.SENDFILE)"""
    destination_id: int
    context: int
    file_path: str
    qos: int
    
    @classmethod
    def from_raw(cls, header: MessageHeader, msg_body: Dict) -> "FileReqMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            destination_id=msg_body["did"],
            context=msg_body["context"],
            file_path=msg_body["file"],
            qos=msg_body["rl"]
        )

@dataclass
class FileFinMessage(Message):
    """文件传输完成 (MID.SENDFIN)"""
    context: int
    file_path: str
    
    @classmethod
    def from_raw(cls, header: MessageHeader, msg_body: Dict) -> "FileFinMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            context=msg_body["context"],
            file_path=msg_body["file"]
        )

@dataclass
class FileRecvMessage(Message):
    """文件接收通知 (MID.RECVFILE)"""
    origin_id: str
    context: int
    file_path: str
    
    @classmethod
    def from_raw(cls, header: MessageHeader, msg_body: Dict) -> "FileRecvMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            origin_id=msg_body["oid"],
            context=msg_body["context"],
            file_path=msg_body["file"]
        )