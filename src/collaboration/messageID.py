from enum import IntEnum

# Message ID
class MessageID(IntEnum):
    """
        协议消息ID枚举
    """
    ACK = 00            # 消息确认

    # 控制接口
    APPREG = 1          # 应用注册
    APPRSP = 2          # 注册响应

    BROCASTECHO = 10    # 广播推送
    BROCASTPUB = 11     # 广播推送
    BROCASTSUB = 12     # 广播订购
    # BROCASTSUBNTY = 13  # 广播订购通知

    MULTIPUB = 16
    MULTISUB = 17

    PUBLISH = 21
    SUBSCRIBE = 22      # 能力订购
    NOTIFY = 23         # 订购通知
    
    # 流传输
    SENDREQ = 101       # 流发送请求
    SENDRDY = 102       # 流发送就绪
    RECVRDY = 103       # 流接收就绪
    SEND = 104          # 流数据发送
    RECV = 105          # 流数据接收
    SENDEND = 106       # 流发送结束
    RECVEND = 107       # 流接收结束
    
    # 文件传输
    SENDFILE = 111      # 文件发送请求
    SENDFIN = 112       # 文件发送完成
    RECVFILE = 113      # 文件接收通知

    @classmethod
    def get_direction(cls, msg_id):
        """判断消息流向 (返回: '应用->控制层'/'控制层->应用'/'双向')"""
        bidirectional = {cls.BROCASTPUB, cls.BROCASTSUB,
                         # cls.BROCASTSUBNTY,
                         cls.PUBLISH, cls.SUBSCRIBE, cls.NOTIFY}
        if msg_id in bidirectional:
            return "双向"
            
        to_control = {cls.APPREG,
                      cls.MULTIPUB, cls.MULTISUB,
                      cls.SENDREQ, cls.SEND, cls.SENDEND,
                      cls.SENDFILE}
        return "应用->控制层" if msg_id in to_control else "控制层->应用"
    
    @classmethod
    def is_control(cls, msg_id) -> bool:
        control_mids = {
            cls.APPREG, cls.APPRSP,
            cls.BROCASTPUB, cls.BROCASTSUB,
            # cls.BROCASTSUBNTY,
            cls.PUBLISH, cls.SUBSCRIBE, cls.NOTIFY
        }
        # 兼容直接传入整数值的情况
        if isinstance(msg_id, MessageID):
            mid_value = msg_id.value
        else:
            mid_value = msg_id
            
        if mid_value in {m.value for m in control_mids}:
            return True
            
        return False

    @classmethod
    def get_name(cls, msg_id: int) -> str:
        """根据消息ID获取枚举名称字符串"""
        # 处理枚举实例输入
        if isinstance(msg_id, cls):
            return msg_id.name
        
        # 整数值匹配
        for member in cls:
            if member.value == msg_id:
                return member.name
        return "UNKNOWN"


