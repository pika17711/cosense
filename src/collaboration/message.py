from dataclasses import dataclass
from enum import IntEnum
import logging
from typing import Optional, Type, Dict, Any
import json
from datetime import datetime
import zmq
from config import AppConfig
from collaboration.messageID import MessageID

"""
接收到的消息
"""


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
    
    @classmethod
    def from_dict(cls, data: Dict) -> "MessageHeader":
        try:
            return cls(
                mid=MessageID(data["mid"]),
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
        header = MessageHeader.from_dict(raw_data)
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
            MessageID.SENDREQ: SendReqMessage,
            MessageID.SENDRDY: SendRdyMessage,
            MessageID.RECVRDY: SendRdyMessage,
            MessageID.SEND: SendMessage,
            MessageID.RECV: SendMessage,
            MessageID.SENDEND: SendEndMessage,
            MessageID.RECVEND: SendEndMessage,
            MessageID.SENDFILE: SendFileMessage,
            MessageID.SENDFIN: SendFinMessage,
            MessageID.RECVFILE: RecvFileMessage,
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
            oid=msg_body["oid"],
            topic=msg_body["topic"],
            coopmap=msg_body["coopmap"],
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
            coopmap=msg_body["coopmap"],
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


class SubscribeAct(IntEnum):
    FIN = 0
    ACKUPD = 1

@dataclass
class SubscribeMessage(Message):
    """能力订购 (MID.SUBSCRIBE)"""
    oid: int
    # did 无
    topic: int
    act: SubscribeAct
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
            coopmap=msg_body["coopmap"],
            bearinfo=bool(msg_body.get("bearinfo", 0))
        )

class NotifyAct(IntEnum):
    NTY = 0
    ACK = 1
    FIN = 2

@dataclass
class NotifyMessage(Message):
    """订购通知 (MID.NOTIFY)"""
    oid: int
    # did 无
    topic: int
    act: NotifyAct
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
            coopmap=msg_body["coopmap"],
            bearcap=msg_body.get("bearcap")
        )

# ----------------------------
# 流传输消息实现
# ----------------------------
@dataclass
class SendReqMessage(Message):
    """流发送请求 (MID.SENDREQ)"""
    did: AppConfig.id_t
    context: AppConfig.cid_t
    rl: int
    pt: int
    aoi: int
    mode: int

    @classmethod
    def from_raw(cls, header: MessageHeader, msg_body: Dict) -> "SendReqMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            did=msg_body["did"],
            context=msg_body["context"],
            rl=msg_body["rl"],
            pt=msg_body["pt"],
            aoi=msg_body["aoi"],
            mode=msg_body["mode"]
        )

@dataclass
class SendRdyMessage(Message):
    did: AppConfig.id_t
    context: AppConfig.cid_t
    sid: AppConfig.sid_t
    
    @classmethod
    def from_raw(cls, header: MessageHeader, msg_body: Dict) -> "SendRdyMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            did=msg_body["did"],
            context=msg_body["context"],
            sid=msg_body["sid"],
        )

@dataclass
class RecvRdyMessage(Message):
    oid: AppConfig.id_t
    context: AppConfig.cid_t
    sid: AppConfig.sid_t

    @classmethod
    def from_raw(cls, header: MessageHeader, msg_body: Dict) -> "RecvRdyMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            oid=msg_body["oid"],
            context=msg_body["context"],
            sid=msg_body["sid"],
        )

@dataclass
class SendMessage(Message):
    sid: AppConfig.id_t
    data: bytes

    @classmethod
    def from_raw(cls, header: MessageHeader, msg_body: Dict) -> "SendMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            sid=msg_body["sid"],
            data=msg_body["data"]
        )

@dataclass
class RecvMessage(Message):
    sid: AppConfig.id_t
    data: bytes

    @classmethod
    def from_raw(cls, header: MessageHeader, msg_body: Dict) -> "RecvMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            sid=msg_body["sid"],
            data=msg_body["data"]
        )

@dataclass
class SendEndMessage(Message):
    sid: int

    @classmethod
    def from_raw(cls, header: MessageHeader, msg_body: Dict) -> "SendEndMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            stream_id=msg_body["sid"],
            context=msg_body["context"]
        )

@dataclass
class RecvEndMessage(Message):
    sid: int

    @classmethod
    def from_raw(cls, header: MessageHeader, msg_body: Dict) -> "RecvEndMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            stream_id=msg_body["sid"],
        )

# ----------------------------
# 文件传输消息实现
# ----------------------------
@dataclass
class SendFileMessage(Message):
    """文件发送请求 (MID.SENDFILE)"""
    did: AppConfig.id_t
    context: AppConfig.cid_t
    rl: int
    pt: int
    file: str
    
    @classmethod
    def from_raw(cls, header: MessageHeader, msg_body: Dict) -> "SendFileMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            destination_id=msg_body["did"],
            context=msg_body["context"],
            rl=msg_body["rl"],
            pt=msg_body["pt"],
            file=msg_body["file"],
        )

@dataclass
class SendFinMessage(Message):
    """文件传输完成 (MID.SENDFIN)"""
    did: AppConfig.id_t
    context: int
    file: str
    
    @classmethod
    def from_raw(cls, header: MessageHeader, msg_body: Dict) -> "SendFinMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            did=msg_body["did"],
            context=msg_body["context"],
            file=msg_body["file"]
        )

@dataclass
class RecvFileMessage(Message):
    """文件接收通知 (MID.RECVFILE)"""
    oid: str
    context: int
    file: str
    
    @classmethod
    def from_raw(cls, header: MessageHeader, msg_body: Dict) -> "RecvFileMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            origin_id=msg_body["oid"],
            context=msg_body["context"],
            file_path=msg_body["file"]
        )