from dataclasses import asdict, dataclass
from enum import IntEnum
import logging
from typing import Optional, Type, Dict, Any, List
import json
from datetime import datetime
import appType
import zmq
from appConfig import AppConfig
from collaboration.messageID import MessageID
from utils.common import base64_decode

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

    def from_raw(self, header, raw):
        pass

    @classmethod
    def parse(cls, raw_data: Dict) -> "Message":
        """工厂方法：从原始字典创建具体消息对象"""
        header = MessageHeader.from_dict(raw_data)
        msg_class = cls._get_message_class(header.mid)
        return msg_class.from_raw(header, raw_data) # type: ignore

    def __str__(self) -> str:
        data = asdict(self)
        def process_value(v):
            return f'{len(v)}B binary data' if isinstance(v, bytes) else v
        data = {k: process_value(v) for k, v in data.items()}
        data["header"] = asdict(self.header)  # 嵌套结构展开
        data["header"]["mid"] = MessageID(data["header"]["mid"]).name
        return '\n' + json.dumps(data, indent=2, ensure_ascii=False)

    @classmethod
    def _get_message_class(cls, mid: MessageID) -> "Message":
        mapping = {
            MessageID.APPREG: AppRegMessage,
            MessageID.APPRSP: AppRspMessage,
            MessageID.BROCASTPUB: BroadcastPubMessage,
            MessageID.BROCASTSUB: BroadcastSubMessage,
            # MessageID.BROCASTSUBNTY: BroadcastSubNtyMessage,
            MessageID.PUBLISH: PublishMessage,
            MessageID.SUBSCRIBE: SubscribeMessage,
            MessageID.NOTIFY: NotifyMessage,
            MessageID.SENDREQ: SendReqMessage,
            MessageID.SENDRDY: SendRdyMessage,
            MessageID.RECVRDY: RecvRdyMessage,
            MessageID.SEND: SendMessage,
            MessageID.RECV: RecvMessage,
            MessageID.SENDEND: SendEndMessage,
            MessageID.RECVEND: RecvEndMessage,
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
    oid: appType.id_t
    topic: str
    cseq: int
    payload: List
    extra: Optional[dict] = None

    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "BroadcastPubMessage":
        msg_body = raw['msg']

        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            oid=msg_body["oid"],
            topic=msg_body["topic"],
            cseq=msg_body['cseq'],
            payload=msg_body['payload'],
            extra=msg_body.get("extra")
        )

@dataclass
class BroadcastSubMessage(Message):
    """广播订购 (MID.BROCASTSUB)"""
    oid: appType.id_t
    topic: int
    context: appType.cid_t
    cseq: int
    payload: List
    
    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "BroadcastSubMessage":
        msg_body = raw['msg']

        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            oid=msg_body["oid"],
            topic=msg_body["topic"],
            context=msg_body["context"],
            cseq=msg_body['cseq'],
            payload=msg_body['payload']
        )

# @dataclass
# class BroadcastSubNtyMessage(Message):
#     """广播订购通知 (MID.BROCASTSUBNTY)"""
#     oid: appType.id_t
#     topic: int
#     context: appType.cid_t
#     coopmap: bytes
#     coopmaptype: int
#     bearcap: int
#
#     @classmethod
#     def from_raw(cls, header: MessageHeader, raw: Dict) -> "BroadcastSubNtyMessage":
#         msg_body = raw['msg']
#         coopmap = msg_body.get("coopmap")
#         if coopmap == None:
#             coopmap = msg_body.get("coopMap")
#         if coopmap == None:
#             raise KeyError('coopmap')
#
#         coopmaptype = msg_body.get("coopmaptype")
#         if coopmaptype == None:
#             coopmaptype = msg_body.get("coopMapType", 1)
#
#         return cls(
#             header=header,
#             direction=MessageID.get_direction(header.mid),
#             oid=msg_body["oid"],
#             topic=msg_body["topic"],
#             context=msg_body["context"],
#             coopmap=coopmap,
#             coopmaptype=coopmaptype,
#             bearcap=msg_body["bearcap"]
#         )


class PublishAct(IntEnum):
    NTY = 0
    ACK = 1
    FIN = 2


@dataclass
class PublishMessage(Message):
    oid: appType.id_t
    did: appType.id_t
    topic: str
    context: appType.cid_t
    cseq: int
    act: PublishAct
    payload: List

    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "PublishMessage":
        msg_body = raw['msg']

        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            oid=msg_body["oid"],
            did=msg_body['did'],
            topic=msg_body["topic"],
            context=msg_body["context"],
            cseq=msg_body['cseq'],
            act=PublishAct(msg_body["act"]),
            payload=msg_body['payload']
        )


class SubscribeAct(IntEnum):
    FIN = 0
    ACKUPD = 1
    # UPD = 2


@dataclass
class SubscribeMessage(Message):
    """能力订购 (MID.SUBSCRIBE)"""
    oid: appType.id_t
    did: appType.id_t
    topic: str
    act: SubscribeAct
    context: appType.cid_t
    cseq: int
    payload: List
    
    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "SubscribeMessage":
        msg_body = raw['msg']

        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            oid=msg_body["oid"],
            did=msg_body['did'],
            topic=msg_body["topic"],
            act=SubscribeAct(msg_body["act"]),
            context=msg_body["context"],
            cseq=msg_body['cseq'],
            payload=msg_body['payload']
        )


class NotifyAct(IntEnum):
    NTY = 0
    ACK = 1
    FIN = 2


@dataclass
class NotifyMessage(Message):
    """订购通知 (MID.NOTIFY)"""
    oid: appType.id_t
    did: appType.id_t
    topic: str
    context: appType.cid_t
    cseq: int
    act: NotifyAct
    payload: List
    
    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "NotifyMessage":
        msg_body = raw['msg']

        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            oid=msg_body["oid"],
            did=msg_body['did'],
            topic=msg_body["topic"],
            context=msg_body["context"],
            cseq=msg_body['cseq'],
            act=NotifyAct(msg_body["act"]),
            payload=msg_body['payload']
        )

# ----------------------------
# 流传输消息实现
# ----------------------------
@dataclass
class SendReqMessage(Message):
    """流发送请求 (MID.SENDREQ)"""
    context: appType.cid_t
    did: appType.id_t
    sid: appType.sid_t
    rl: int
    pt: int
    aoi: int
    mode: int
    ip: Optional[str] = ''
    port: Optional[int] = 0
    ip2: Optional[str] = ''
    port2: Optional[int] = 0

    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "SendReqMessage":
        msg_body = raw['msg']

        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            context=msg_body["context"],
            did=msg_body["did"],
            sid=msg_body['sid'],
            rl=msg_body["rl"],
            pt=msg_body["pt"],
            aoi=msg_body["aoi"],
            mode=msg_body["mode"],
            ip=msg_body.get('ip', ''),
            port=msg_body.get('port', 0),
            ip2=msg_body.get('ip2', ''),
            port2=msg_body.get('port2', 0),
        )


@dataclass
class SendRdyMessage(Message):
    did: appType.id_t
    context: appType.cid_t
    sid: appType.sid_t
    
    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "SendRdyMessage":
        msg_body = raw['msg']

        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            did=msg_body["did"],
            context=msg_body["context"],
            sid=msg_body["sid"]
        )


@dataclass
class RecvRdyMessage(Message):
    oid: appType.id_t
    context: appType.cid_t
    sid: appType.sid_t

    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "RecvRdyMessage":
        msg_body = raw['msg']

        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            oid=msg_body["oid"],
            context=msg_body["context"],
            sid=msg_body["sid"]
        )


@dataclass
class SendMessage(Message):
    context: appType.cid_t
    did: appType.id_t
    sid: appType.sid_t
    data: bytes

    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "SendMessage":
        msg_body = raw['msg']

        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            context=msg_body['context'],
            did=msg_body['did'],
            sid=msg_body["sid"],
            data=msg_body["data"]
        )


@dataclass
class RecvMessage(Message):
    context: appType.cid_t
    did: appType.id_t
    sid: appType.sid_t
    data: bytes

    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "RecvMessage":
        msg_body = raw['msg']

        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            context=msg_body['context'],
            did=msg_body['did'],
            sid=msg_body["sid"],
            data=msg_body["data"]
        )


@dataclass
class SendEndMessage(Message):
    context: appType.cid_t
    did: appType.id_t
    sid: appType.sid_t

    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "SendEndMessage":
        msg_body = raw['msg']

        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            context=msg_body['context'],
            did=msg_body['did'],
            sid=msg_body["sid"]
        )


@dataclass
class RecvEndMessage(Message):
    context: appType.cid_t
    did: appType.id_t
    sid: appType.sid_t

    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "RecvEndMessage":
        msg_body = raw['msg']

        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            context=msg_body['context'],
            did=msg_body['did'],
            sid=msg_body["sid"]
        )

# ----------------------------
# 文件传输消息实现
# ----------------------------
@dataclass
class SendFileMessage(Message):
    """文件发送请求 (MID.SENDFILE)"""
    context: appType.cid_t
    did: appType.id_t
    rl: int
    pt: int
    file: str
    
    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "SendFileMessage":
        msg_body = raw['msg']

        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            context=msg_body['context'],
            did=msg_body['did'],
            rl=msg_body['rl'],
            pt=msg_body['pt'],
            file=msg_body['file']
        )


@dataclass
class SendFinMessage(Message):
    """文件传输完成 (MID.SENDFIN)"""
    context: appType.cid_t
    did: appType.id_t
    file: str
    
    @classmethod
    def from_raw(cls, header: MessageHeader, raw: Dict) -> "SendFinMessage":
        msg_body = raw['msg']

        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            context=msg_body["context"],
            did=msg_body['did'],
            file=msg_body["file"]
        )


@dataclass
class RecvFileMessage(Message):
    """文件接收通知 (MID.RECVFILE)"""
    context: appType.cid_t
    oid: appType.id_t
    file: str
    
    @classmethod
    def from_raw(cls, header:   MessageHeader, raw: Dict) -> "RecvFileMessage":
        msg_body = raw['msg']

        return cls(
            header=header,
            direction=MessageID.get_direction(header.mid),
            context=msg_body["context"],
            oid=msg_body["oid"],
            file=msg_body["file"]
        )