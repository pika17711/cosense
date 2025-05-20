from dataclasses import dataclass
from enum import IntEnum
import logging
from typing import Optional, Type, Dict, Any
import json
from datetime import datetime
import appType
import zmq
from appConfig import AppConfig
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
    tid: Optional[int]
    
    @classmethod
    def from_dict(cls, data: Dict) -> "MessageHeader":
        try:
            return cls(
                mid=MessageID(data["mid"]),
                tid=data.get("tid")
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
        return msg_class.from_raw(header, raw_data)

    @classmethod
    def _get_message_class(cls, mid: MessageID) -> "Message":
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
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "AckMessage":
        msg_body = raw['msg']
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
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "AppRegMessage":
        msg_body = raw['msg']
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
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "AppRspMessage":
        msg_body = raw['msg']
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
    coopmaptype: int
    extra: Optional[dict] = None

    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "BroadcastPubMessage":
        msg_body = raw['msg']
        coopmap = msg_body.get("coopmap")
        if coopmap == None:
            coopmap = msg_body.get("coopMap")
        if coopmap == None:
            raise KeyError('coopmap')

        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            oid=msg_body["oid"],
            topic=msg_body["topic"],
            coopmap=coopmap,
            coopmaptype=msg_body.get("coopmaptype", 1),
            extra=msg_body.get("extra")
        )

@dataclass
class BroadcastSubMessage(Message):
    """广播订购 (MID.BROCASTSUB)"""
    oid: appType.id_t
    topic: int
    context: appType.cid_t
    coopmap: bytes
    coopmaptype: int
    bearcap: int
    
    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "BroadcastSubMessage":
        msg_body = raw['msg']
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            oid=msg_body["oid"],
            topic=msg_body["topic"],
            context=msg_body["context"],
            coopmap=msg_body["coopmap"],
            coopmaptype=msg_body.get("coopmaptype", 1),
            bearcap=msg_body["bearcap"]
        )

@dataclass
class BroadcastSubNtyMessage(Message):
    """广播订购通知 (MID.BROCASTSUBNTY)"""
    oid: appType.id_t
    topic: int
    context: appType.cid_t
    coopmap: bytes
    coopmaptype: int
    bearcap: int
    
    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "BroadcastSubNtyMessage":
        msg_body = raw['msg']
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            oid=msg_body["oid"],
            topic=msg_body["topic"],
            context=msg_body["context"],
            coopmap=msg_body["coopmap"],
            coopmaptype=msg_body.get("coopmaptype", 1),
            bearcap=msg_body["bearcap"]
        )


class SubscribeAct(IntEnum):
    FIN = 0
    ACKUPD = 1

@dataclass
class SubscribeMessage(Message):
    """能力订购 (MID.SUBSCRIBE)"""
    oid: appType.id_t
    topic: int
    act: SubscribeAct
    context: appType.cid_t
    coopmap: bytes
    coopmaptype: int
    bearcap: int
    
    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "SubscribeMessage":
        msg_body = raw['msg']
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            oid=msg_body["oid"],
            topic=msg_body["topic"],
            act=SubscribeAct(msg_body["act"]),
            context=msg_body["context"],
            coopmap=msg_body["coopmap"],
            coopmaptype=msg_body.get("coopmaptype", 1),
            bearcap=msg_body.get("bearcap", 0)
        )

class NotifyAct(IntEnum):
    NTY = 0
    ACK = 1
    FIN = 2

@dataclass
class NotifyMessage(Message):
    """订购通知 (MID.NOTIFY)"""
    oid: appType.id_t
    # did 无
    topic: int
    act: NotifyAct
    context: appType.cid_t
    coopmap: bytes
    coopmaptype: int
    bearcap: Optional[int]
    
    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "NotifyMessage":
        msg_body = raw['msg']
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            oid=msg_body["oid"],
            topic=msg_body["topic"],
            act=NotifyAct(msg_body["act"]),
            context=msg_body["context"],
            coopmap=msg_body["coopmap"],
            coopmaptype=msg_body.get("coopmaptype", 1),
            bearcap=msg_body.get("bearcap")
        )

# ----------------------------
# 流传输消息实现
# ----------------------------
@dataclass
class SendReqMessage(Message):
    """流发送请求 (MID.SENDREQ)"""
    did: appType.id_t
    context: appType.cid_t
    rl: int
    pt: int
    aoi: int
    mode: int

    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "SendReqMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            did=raw["did"],
            context=raw["context"],
            rl=raw["rl"],
            pt=raw["pt"],
            aoi=raw["aoi"],
            mode=raw["mode"]
        )

@dataclass
class SendRdyMessage(Message):
    did: appType.id_t
    context: appType.cid_t
    sid: appType.sid_t
    
    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "SendRdyMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            did=raw["did"],
            context=raw.get("context"),
            sid=raw["sid"],
        )

@dataclass
class RecvRdyMessage(Message):
    oid: appType.id_t
    context: appType.cid_t
    sid: appType.sid_t

    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "RecvRdyMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            oid=raw["oid"],
            context=raw["context"],
            sid=raw["sid"],
        )

@dataclass
class SendMessage(Message):
    sid: appType.id_t
    data: bytes

    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "SendMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            sid=raw["sid"],
            data=raw["data"]
        )

@dataclass
class RecvMessage(Message):
    sid: appType.id_t
    data: bytes

    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "RecvMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            sid=raw["sid"],
            data=raw["data"]
        )

@dataclass
class SendEndMessage(Message):
    sid: int
    context: Optional[appType.cid_t]

    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "SendEndMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            sid=raw["sid"],
            context=raw.get("context")
        )

@dataclass
class RecvEndMessage(Message):
    sid: appType.sid_t

    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "RecvEndMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            sid=raw["sid"],
        )

# ----------------------------
# 文件传输消息实现
# ----------------------------
@dataclass
class SendFileMessage(Message):
    """文件发送请求 (MID.SENDFILE)"""
    did: appType.id_t
    context: appType.cid_t
    rl: int
    pt: int
    file: str
    
    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "SendFileMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            did=raw["did"],
            context=raw["context"],
            rl=raw["rl"],
            pt=raw["pt"],
            file=raw["file"],
        )

@dataclass
class SendFinMessage(Message):
    """文件传输完成 (MID.SENDFIN)"""
    did: appType.id_t
    context: int
    file: str
    
    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "SendFinMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            did=raw["did"],
            context=raw["context"],
            file=raw["file"]
        )

@dataclass
class RecvFileMessage(Message):
    """文件接收通知 (MID.RECVFILE)"""
    oid: str
    context: int
    file: str
    
    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "RecvFileMessage":
        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            oid=raw["oid"],
            context=raw["context"],
            file=raw["file"]
        )