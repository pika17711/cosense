from __future__ import annotations

from functools import wraps
import logging
from typing import List, Union
import appType
from collaboration.broadcastCollaborationContext import BCContext, BCContextState
from collaboration.collaborationContext import CContext, CContextCoteeState, CContextCotorState, CSContextCoteeState, CSContextCotorState
from collaboration.collaborationTable import CollaborationTable
from collaboration.contextGenerator import ContextGenerator
from collaboration.coopMap import CoopMap, CoopMapType
from collaboration.message import BroadcastPubMessage, BroadcastSubMessage, BroadcastSubNtyMessage, Message, NotifyAct, NotifyMessage, RecvEndMessage, RecvFileMessage, RecvMessage, RecvRdyMessage, SendFinMessage, SendRdyMessage, SubscribeAct, SubscribeMessage
from collaboration.transactionHandler import transactionHandler
from appConfig import AppConfig
from perception.perceptionRPCClient import PerceptionRPCClient
from detection.detectionRPCClient import DetectionRPCClient
import numpy as np
from utils import InfoDTO
from utils.common import read_binary_file, server_assert, server_logic_error, server_not_implemented

def ContextStateTransition(name):
    """状态转移函数的装饰器，提供自动加锁和日志记录功能"""
    def get_state(ctx: Union[CContext, BCContext]):
        state = None
        if name == 'cctx':
            state = ctx.state
        elif name == 'bcctx':
            state = ctx.state
        elif name == 'stream':
            server_assert(isinstance(ctx, CContext))
            state = ctx.stream_state # type: ignore
        else:
            server_assert(False)
        return state

    def decorator(func):
        @wraps(func)  # 保留原函数元信息
        def wrapper(self, ctx: Union[CContext, BCContext], *args, **kwargs):
            # 获取上下文锁
            with ctx.lock:
                old_state = get_state(ctx)
                func_name = func.__name__
                logging.info(f"执行状态转移: {func_name}, 上下文ID: {ctx.cid}, "
                            f"原状态: {old_state}")
                try:
                    # 执行实际的状态转移
                    result = func(self, ctx, *args, **kwargs)
                    # 记录状态转移成功
                    new_state = get_state(ctx)
                    if old_state != new_state:
                        logging.info(f"状态转移成功: 上下文ID: {ctx.cid}, "
                                    f"{old_state} -> {new_state}")
                    else:
                        logging.warning(f"状态未变更: 上下文ID: {ctx.cid}, "
                                        f"函数: {func_name}")
                    return result
                except Exception as e:
                    # 记录状态转移异常
                    logging.error(f"状态转移失败: 上下文ID: {ctx.cid}, "
                                f"原状态: {old_state}, 错误: {str(e)}")
                    raise  # 重新抛出异常，不掩盖错误
        return wrapper
    return decorator

class CollaborationService:
    """
        业务逻辑类，包含：
            1. 发送消息的逻辑
            2. 接收消息的逻辑
            3. context状态转移的逻辑
            4. 一些工具函数和封装函数
    """

    def __init__(self, 
                 cfg: AppConfig,
                 ctable: 'CollaborationTable',
                 perception_client: 'PerceptionRPCClient',
                 detection_client: 'DetectionRPCClient',
                 tx_handler: 'transactionHandler'
                 ) -> None:
        self.cfg = cfg 
        self.cid_gen = ContextGenerator(cfg)
        self.ctable = ctable
        self.perception_client = perception_client
        self.detection_client = detection_client
        self.tx_handler = tx_handler

    @ContextStateTransition('cctx')
    def cctx_to_waitnty(self, cctx: CContext):
        server_assert(cctx.is_cotee(), "上下文角色必须是被协作者")
        if cctx.state != CContextCoteeState.PENDING:
            return

        cctx.update_active()
        cctx.state = CContextCoteeState.WAITNTY
        self.ctable.add_cctx(cctx)
        self.ctable.add_waitnty(cctx)

    @ContextStateTransition('cctx')
    def cctx_to_subscribing(self, cctx: CContext):
        server_assert(cctx.is_cotee(), "上下文角色必须是被协作者")
        if cctx.state != CContextCoteeState.WAITNTY:
            return
        cctx.update_active()
        self.ctable.rem_waitnty(cctx)
        logging.debug(f"订阅 {cctx.remote_id()}, context: {cctx.cid}")
        cctx.state = CContextCoteeState.SUBSCRIBING
        self.ctable.add_subscribing(cctx)
        self.stream_to_waitrdy(cctx)

    @ContextStateTransition('cctx')
    def cctx_to_sendnty(self, cctx: CContext):
        server_assert(cctx.is_cotor(), "上下文角色必须是协作者")
        if cctx.state != CContextCotorState.PENDING:
            return
        cctx.update_active()
        cctx.state = CContextCotorState.SENDNTY
        self.ctable.add_sendnty(cctx)
        self.ctable.add_cctx(cctx)

    @ContextStateTransition('cctx')
    def cctx_to_subscribed(self, cctx: CContext, coopmap: CoopMap):
        server_assert(cctx.is_cotor(), "上下文角色必须是协作者")
        if cctx.state != CContextCotorState.SENDNTY:
            return
        cctx.update_active()
        self.ctable.rem_sendnty(cctx)
        logging.debug(f"被 { cctx.remote_id()} 订阅, context: {cctx.cid}")
        cctx.state = CContextCotorState.SUBSCRIBED
        self.ctable.add_subscribed(cctx, coopmap)

    @ContextStateTransition('cctx')
    def cctx_to_closed(self, cctx: CContext):
        if cctx.state == CContextCoteeState.CLOSED or cctx.state == CContextCoteeState.CLOSED:
            return

        if cctx.is_cotor():
            if cctx.state == CContextCotorState.SUBSCRIBED:
                self.ctable.rem_subscribed(cctx)
            cctx.state = CContextCotorState.CLOSED
        else:
            if cctx.state == CContextCoteeState.SUBSCRIBING:
                self.ctable.rem_subscribing(cctx)
            cctx.state = CContextCoteeState.CLOSED
        bcctx = self.ctable.get_bcctx(cctx.cid)
        if bcctx is not None:
            self.bcctx_rem_cctx(bcctx, cctx)
        self.stream_to_end(cctx)
        self.ctable.rem_cctx(cctx)

    @ContextStateTransition('stream')
    def stream_to_waitrdy(self, cctx: CContext):
        server_assert(cctx.is_cotee())
        cctx.update_active()
        if cctx.stream_state != CSContextCoteeState.PENDING:
            return
        cctx.stream_state = CSContextCoteeState.WAITRDY

    @ContextStateTransition('stream')
    def stream_to_recvrdy(self, cctx: CContext):
        server_assert(cctx.is_cotee())
        cctx.update_active()
        if cctx.stream_state != CSContextCoteeState.WAITRDY:
            return
        cctx.stream_state = CSContextCoteeState.RECVRDY
        self.ctable.add_stream(cctx.sid, cctx)

    @ContextStateTransition('stream')
    def stream_to_sendreq(self, cctx: CContext):
        server_assert(cctx.is_cotor())
        cctx.update_active()
        if cctx.stream_state != CSContextCotorState.PENDING:
            return
        cctx.stream_state = CSContextCotorState.SENDREQ

    @ContextStateTransition('stream')
    def stream_to_sendrdy(self, cctx: CContext):
        server_assert(cctx.is_cotor())
        cctx.update_active()
        if cctx.stream_state != CSContextCotorState.SENDREQ:
            return
        self.ctable.add_stream(cctx.sid, cctx)
        cctx.stream_state = CSContextCotorState.SENDRDY

    @ContextStateTransition('stream')
    def stream_to_end(self, cctx: CContext):
        if cctx.stream_state == CSContextCotorState.SENDEND or cctx.stream_state == CSContextCoteeState.RECVEND:
            return

        if cctx.is_cotor():
            cctx.stream_state = CSContextCotorState.SENDEND
        else:
            cctx.stream_state = CSContextCoteeState.RECVEND
        if cctx.have_sid():
            self.ctable.rem_stream(cctx.sid)

    @ContextStateTransition('bcctx')
    def bcctx_to_waitbnnty(self, bcctx: BCContext):
        if bcctx.state != BCContextState.PENDING:
            return
        bcctx.state = BCContextState.WAITBNTY
        self.ctable.add_bcctx(bcctx)

    @ContextStateTransition('bcctx')
    def bcctx_to_closed(self, bcctx: BCContext):
        if bcctx.state == BCContextState.PENDING:
            logging.warning(f"广播会话 {bcctx.cid}未发送广播订阅消息即被关闭")

        self.ctable.rem_bcctx(bcctx)
        bcctx.state = BCContextState.CLOSED

    def bcctx_add_cctx(self, bcctx: BCContext, cctx: CContext):
        bcctx.cctx_set.add(cctx)

    def bcctx_rem_cctx(self, bcctx: BCContext, cctx: CContext):
        bcctx.cctx_set.remove(cctx)

    def get_self_coopmap(self, coopmap_type: CoopMapType = CoopMapType.CommMask):
        if coopmap_type == CoopMapType.DEBUG:
            return CoopMap(self.cfg.id, coopmap_type, None, None)

        comm_mask, _, lidar_pose, _ = self.detection_client.get_comm_mask_and_lidar_pose()
        if coopmap_type == CoopMapType.Empty:
            return CoopMap(self.cfg.id, coopmap_type, None, lidar_pose)
        if coopmap_type == CoopMapType.CommMask:
            return CoopMap(self.cfg.id, coopmap_type, comm_mask, lidar_pose)
        if coopmap_type == CoopMapType.RequestMap:
            request_map = 1 - comm_mask
            return CoopMap(self.cfg.id, coopmap_type, request_map, lidar_pose)

    def check_need_broadcastpub(self, msg: BroadcastPubMessage):
        """
            是否需要remote的广播推送
            1. 如果当前已经建立了local被协作的连接，则不需要再次由广播推送建立连接
                local: cotee
                remote: cotor
            2. 判断协作图的重叠是否超过阈值
        """
        if msg.oid == self.cfg.id:
            return False

        if self.ctable.get_waitnty_by_id(msg.oid) is not None or self.ctable.get_subscribing_by_id(msg.oid) is not None:
            return False
        if self.cfg.collaboration_debug:
            return True
        coopmap = CoopMap.deserialize(msg.coopmap)
        if coopmap == None:
            return False
        coopmap_self = self.get_self_coopmap(CoopMapType.CommMask)
        ratio = CoopMap.calculate_overlap_ratio(coopmap, coopmap_self)
        return ratio >= self.cfg.overlap_threshold

    def check_need_broadcastsubnty(self, bcctx: BCContext, msg: BroadcastSubNtyMessage):
        """
            是否接收remote的广播推送通知
                这里一定是因为local发送了broadcastsub
        """
        if self.ctable.get_waitnty_by_id(msg.oid) is not None or self.ctable.get_subscribing_by_id(msg.oid) is not None:
            return False
        coopmap = CoopMap.deserialize(msg.coopmap)
        if coopmap == None:
            return False
        coopmap_self = self.get_self_coopmap(CoopMapType.CommMask)
        ratio = CoopMap.calculate_overlap_ratio(coopmap, coopmap_self)
        return ratio >= self.cfg.overlap_threshold

    def check_need_subscribe(self, msg: SubscribeMessage):
        """
            是否接受remote的订阅
            目前开启订阅的操作只有两种:
            1. broadcastpub
            2. broadcastsub
            这两种都已经检查过了重叠率, 所以无需再检查重叠率
            此时要求协作图类型必须为请求图
        """
        return msg.coopmaptype == CoopMapType.RequestMap

    def check_need_broadcastsub(self, msg: BroadcastSubMessage):
        """
            是否接收remote的广播订阅
            1. 如果当前已经建立了local协作的连接，则不需要再次由广播推送建立连接
                local: cotor
                remote: cotee
            2. 判断协作图的重叠是否超过阈值
        """
        if msg.oid == self.cfg.id:
            return False

        if self.ctable.get_sendnty_by_id(msg.oid) is not None or self.ctable.get_subscribed_by_id(msg.oid):
            return False
        coopmap = CoopMap.deserialize(msg.coopmap)
        if coopmap == None:
            return False
        coopmap_self = self.get_self_coopmap(CoopMapType.CommMask)
        ratio = CoopMap.calculate_overlap_ratio(coopmap, coopmap_self)
        return ratio >= self.cfg.overlap_threshold

    def broadcastsub(self):
        """
            广播订阅
                local开启一个新广播会话bcctx，启动一个新协程bcctx_loop(bcctx)，进行消息接收和处理
        """
        coopMapType = CoopMapType.CommMask
        cid = self.cid_gen()
        coopmap = self.get_self_coopmap(coopMapType)
        decoopmap = CoopMap.serialize(coopmap)
        bearCap = 1
        bcctx = BCContext(self.cfg, cid)
        with bcctx.lock:
            self.tx_handler.brocastsub(self.cfg.id, self.cfg.topic, cid, decoopmap, coopMapType, bearCap)
            self.bcctx_to_waitbnnty(bcctx)

    def broadcastpub_send(self):
        """
            广播推送发送
        """
        coopMapType = CoopMapType.CommMask
        coopmap = self.get_self_coopmap(coopMapType)
        decoopmap = CoopMap.serialize(coopmap)
        self.tx_handler.brocastpub(self.cfg.id, self.cfg.topic, decoopmap, coopMapType)
    
    def subscribe(self, did: appType.id_t):
        """
            封装subscribe_send
        """
        cid = self.cid_gen()
        cctx = CContext(self.cfg, cid, did, self.cfg.id)
        with cctx.lock:
            self.subscribe_send(cctx, SubscribeAct.ACK)
            self.cctx_to_waitnty(cctx)

    def subscribe_send(self, cctx: CContext, act:SubscribeAct=SubscribeAct.ACK):
        """
            发送订阅
            描述：
                1. act = SubscribeAct.ACKUPD
                    订阅请求
                    local开启一个新会话cctx
                    启动一个新协程cctx_loop(cctx)，进行消息接收和处理
                2. act = SubscribeAct.FIN
                    订阅关闭
                    local是cotee，remote是cotor，不然就是代码逻辑错误

            参数：
                1. cctx
                    对应的context
                2. act
                    订阅请求或订阅关闭
        """
        if act == SubscribeAct.ACK:
            coopMapType = CoopMapType.RequestMap
            coopmap = self.get_self_coopmap(coopMapType)
            decoopmap = CoopMap.serialize(coopmap)
            bearCap = 1
            self.tx_handler.subscribe(self.cfg.id, [cctx.remote_id()], self.cfg.topic, act, cctx.cid, decoopmap, coopMapType, bearCap)
        elif act == SubscribeAct.FIN:
            coopMapType = CoopMapType.Empty
            coopmap = self.get_self_coopmap(coopMapType)
            decoopmap = CoopMap.serialize(coopmap)
            bearCap = 0
            self.tx_handler.subscribe(self.cfg.id, [cctx.remote_id()], self.cfg.topic, act, cctx.cid, decoopmap, coopMapType, bearCap)  # 关闭订阅不需要传协作图
        else:
            server_not_implemented('')

    def notify_send(self, cctx: CContext, act=NotifyAct.ACK):
        """
            通知
            描述：
                1. act = NotifyAct.ACK
                    确认订购
                    对话cctx在收到订阅消息时新建
                    找到对应cctx，发送确认订购后，改变cctx状态
                2. act = NotifyAct.NTY
                    对话内通知
                    未实现
                2. act = NotifyAct.FIN
                    取消订购
                    对话cctx在收到订阅消息时新建
                    找到对应cctx，发送取消订购后，改变cctx状态
            参数：
                1. did
                2. act
        """
        if act == NotifyAct.ACK:
            coopMapType = CoopMapType.Empty
            coopmap = self.get_self_coopmap(coopMapType)
            decoopmap = CoopMap.serialize(coopmap)
            bearCap = 1
            self.tx_handler.notify(self.cfg.id, cctx.remote_id(), self.cfg.topic, act, cctx.cid, decoopmap, coopMapType, bearCap)   # 不需要传协作图
        elif act == NotifyAct.NTY:
            server_not_implemented('')
        elif act == NotifyAct.FIN:
            coopMapType = CoopMapType.Empty
            coopmap = self.get_self_coopmap(coopMapType)
            decoopmap = CoopMap.serialize(coopmap)
            bearCap = 1 
            self.tx_handler.notify(self.cfg.id, cctx.remote_id(), self.cfg.topic, act, cctx.cid, decoopmap, coopMapType, bearCap)   # 不需要传协作图

    def sendreq_send(self, did: appType.id_t, cid: appType.cid_t, rl, pt, aoi, mode):
        self.tx_handler.sendreq(did, cid, rl, pt, aoi, mode)

    def send_send(self, sid: appType.sid_t, data: str):
        self.tx_handler.send(sid, data)

    def sendend_send(self, did: appType.id_t, cid: appType.cid_t, sid: appType.sid_t):
        self.tx_handler.sendend(did, cid, sid)

    def get_stream(self, cctx: CContext):
        server_assert(cctx.is_cotor())
        if cctx.stream_state == CSContextCotorState.PENDING:
            logging.debug(f"context: {cctx.cid} 获取stream")
            rl = 1
            pt = 1
            aoi = 0
            mode = 1
            with cctx.lock:
                self.sendreq_send(cctx.remote_id(), cctx.cid, rl, pt, aoi, mode)
                # self.sendreq_send(cctx.cotor, cctx.cid, rl, pt, aoi, mode)
                self.stream_to_sendreq(cctx)
            cctx.sid_set_event.wait()
            server_assert(cctx.stream_state == CSContextCotorState.SENDRDY)

    def send_data(self, cctx: CContext, data: str):
        if cctx.stream_state == CSContextCotorState.SENDEND:
            logging.debug(f"context: {cctx.cid} 发送结束, 发送数据失败")
            return

        if not cctx.have_sid():
            if cctx.stream_state == CSContextCotorState.SENDREQ:
                logging.debug(f"context: {cctx.cid} 获取stream中, 发送数据失败")
            elif cctx.stream_state == CSContextCotorState.PENDING:
                self.get_stream(cctx)
        else:
            self.send_send(cctx.sid, data) # type: ignore

    def broadcastpub_service(self, msg: BroadcastPubMessage):
        """
            收到广播推送
            1. 判断是否接受
            2. 如果接受，创建cctx，发送订阅请求
        """
        logging.debug(f"APP serve message {msg}")
        need = self.check_need_broadcastpub(msg)
        if need:
            logging.info(f"接收 {msg.oid} 的BROADCASTPUB")
            self.subscribe(msg.oid)
        else:
            logging.info(f"拒绝 {msg.oid} 的BROADCASTPUB")

    def broadcastsub_service(self, msg: BroadcastSubMessage):
        """
            收到广播订阅
            1. 判断是否接受
            2. 如果接受，发送广播订阅通知
        """
        logging.debug(f"APP serve message {msg}")
        need = self.check_need_broadcastsub(msg)
        if need:
            logging.debug(f"接收 {msg.oid} 的BROADCASTSUB")
            coopMapType = CoopMapType.CommMask
            coopmap = self.get_self_coopmap(coopMapType)
            decoopmap = CoopMap.serialize(coopmap)
            bearcap = 1
            # TODO: 这里有一个问题，如果还没有建立会话，就可能对同一个broadcastsub多次发送brocastsubnty
            # , 但broadcastsub发送时间间隔较长, 先忽略这个问题
            self.tx_handler.brocastsubnty(self.cfg.id, msg.oid, self.cfg.topic, msg.context, 
                                            decoopmap, coopMapType, bearcap)
        else:
            logging.debug(f"拒绝 {msg.oid} 的BROADCASTSUB")

    def broadcastsubnty_service(self, msg: BroadcastSubNtyMessage):
        """
            收到广播订阅通知
            1. 寻找广播对话，因为收到了广播订阅，这个广播会话是自身创建的
            2. 检查是否需要对话
            3. 如果需要，创建cctx
        """
        logging.debug(f"APP serve message {msg}")
        bcctx = self.ctable.get_bcctx(msg.context)
        if bcctx == None:
            # 可能是超时或消息发送错误
            logging.warning(f"收到BROADCASTSUBNTY, 广播对话 context:{msg.context} 不存在")
            return
        server_assert(bcctx.state != BCContextState.PENDING)

        if bcctx.state == BCContextState.WAITBNTY:
            need = self.check_need_broadcastsubnty(bcctx, msg)
            if need:
                logging.debug(f"接收 {msg.oid} 的BROADCASTSUBNTY")
                cid = self.cid_gen()
                cctx = CContext(self.cfg, cid, msg.oid, self.cfg.id)
                with cctx.lock:
                    self.subscribe_send(cctx, SubscribeAct.ACK)
                    with bcctx.lock:
                        self.bcctx_add_cctx(bcctx, cctx)
                    self.cctx_to_waitnty(cctx)
            else:
                logging.debug(f"拒绝 {msg.oid} 的BROADCASTSUBNTY")
        elif bcctx.state == BCContextState.CLOSED:
            logging.warning(f"收到BROADCASTSUBNTY, 但对应 context:{msg.context} 已超时")
            pass

    def notify_service(self, msg: NotifyMessage):
        """
            收到notify
            1. 寻找cctx，cctx是local发送订阅通知时创建的，local一定是cotee
            2. 根据消息内容，更新状态
        """
        logging.debug(f"APP serve message {msg}")
        cctx = self.ctable.get_cctx(msg.context, msg.oid, self.cfg.id)
        if cctx == None:
            server_logic_error(f"收到NOTIFY, 但对应 context:{msg.context} 不存在")
            return
        if cctx.is_cotor():
            server_logic_error(f"收到NOTIFY, 但对应 context:{msg.context} 中local是cotor")
            return
        server_assert(cctx.state != CContextCoteeState.PENDING)

        with cctx.lock:
            if cctx.state == CContextCoteeState.WAITNTY:
                if msg.act == NotifyAct.ACK:
                    self.cctx_to_subscribing(cctx)
                elif msg.act == NotifyAct.FIN:
                    self.cctx_to_closed(cctx)
                elif msg.act == NotifyAct.NTY:
                    server_logic_error(f"收到NOTIFY.NTY, 但对应 context:{msg.context} 中状态是WAITNTY, 不应该收到NOTIFY.NTY")
            elif cctx.state == CContextCoteeState.SUBSCRIBING:
                if msg.act == NotifyAct.ACK:
                    server_logic_error(f"收到NOTIFY.ACK, 但对应 context:{msg.context} 中状态是SUBSCRIBING, 不应该收到NOTIFY.ACK")
                elif msg.act == NotifyAct.FIN:
                    self.cctx_to_closed(cctx)
                elif msg.act == NotifyAct.NTY:
                    server_not_implemented(f"收到NOTIFY.ACK, 对应 context:{msg.context} 中状态是SUBSCRIBING")

    def subscribe_ack_service(self, msg: SubscribeMessage):
        """
            收到订阅消息，act=subscribeAct=ACKUPD
            1. 此时自车不应该有cctx（目前只将ACKUPD看作ACK，并未实现UPD）
            2. 创建cctx，自车作为cotor
            3. 判断是否被订阅
            4. 发送notify
        """
        logging.debug(f"APP serve message {msg}")
        cctx = self.ctable.get_cctx(msg.context, self.cfg.id, msg.oid)
        if cctx is not None:
            logging.warning(f"收到SUBSCRIBE, 此时不应该存在此context:{msg.context}")
            return
        cctx = CContext(self.cfg, msg.context, self.cfg.id, msg.oid)
        with cctx.lock:
            self.cctx_to_sendnty(cctx)
            if self.check_need_subscribe(msg):
                self.notify_send(cctx)
                coopmap = CoopMap.deserialize(msg.coopmap)
                self.cctx_to_subscribed(cctx, coopmap)
            else:
                self.cctx_to_closed(cctx)

    def subscribe_fin_service(self, msg: SubscribeMessage):
        """
            收到订阅消息，act=subscribeAct=FIN
            1. 此时自车应该有cctx
            2. 找到cctx
            3. 根据cctx状态行动
        """
        logging.debug(f"APP serve message {msg}")
        cctx = self.ctable.get_cctx(msg.context, self.cfg.id, msg.oid)
        if cctx is None:
            logging.warning(f"收到SUBSCRIBE FIN, 但不存在此context:{msg.context}")
            return
        with cctx.lock:
            if cctx.state == CContextCotorState.PENDING:
                logging.warning(f"收到SUBSCRIBE FIN, 但对应context:{msg.context}还未发送NOTIFY")
            elif cctx.state == CContextCotorState.SENDNTY:
                self.cctx_to_closed(cctx)
            elif cctx.state == CContextCotorState.SUBSCRIBED:
                self.cctx_to_closed(cctx)
            elif cctx.state == CContextCotorState.CLOSED:
                pass

    def subscribe_service(self, msg: SubscribeMessage):
        logging.debug(f"APP serve message {msg}")
        if msg.act == SubscribeAct.ACK:
            self.subscribe_ack_service(msg)
        elif msg.act == SubscribeAct.FIN:
            self.subscribe_fin_service(msg)
        else:
            assert False

    def recvfile_service(self, msg: RecvFileMessage):
        logging.debug(f"APP serve message {msg}")
        cctx = self.ctable.get_cctx_or_panic(msg.context, self.cfg.id, msg.oid)
        with cctx.lock:
            server_assert(cctx.state == CContextCoteeState.SUBSCRIBING)
            data = read_binary_file(msg.file)
            # de_data = InfoDTO.InfoDTOSerializer.deserialize(data)
            de_data = InfoDTO.InfoDTOSerializer.deserialize_from_str(data)
            if de_data is None:
                return
            self.ctable.add_data(de_data)

    def sendfin_service(self, msg: SendFinMessage):
        logging.debug(f"APP serve message {msg}")

    def sendrdy_service(self, msg: SendRdyMessage):
        logging.debug(f"APP serve message {msg}")
        cctx = self.ctable.get_cctx(msg.context, self.cfg.id, msg.did)
        # cctx = self.ctable.get_cctx(msg.context, msg)
        if cctx is None:
            logging.warning(f'sendrdy context不存在: {msg.context}')
            return
        with cctx.lock:
            server_assert(not cctx.have_sid())
            cctx.sid = msg.sid
            self.stream_to_sendrdy(cctx)
            cctx.sid_set_event.set()

    def recvrdy_service(self, msg: RecvRdyMessage):
        logging.debug(f"APP serve message {msg}")
        cctx = self.ctable.get_cctx(msg.context, msg.oid, self.cfg.id)
        if cctx is None:
            logging.warning(f'recvrdy context不存在: {msg.context}')
            return
        cctx.update_active()
        with cctx.lock:
            if cctx.have_sid():
                logging.warning(f'context {msg.context} 已经收到过recvrdy')
                return
            cctx.sid = msg.sid
            self.stream_to_recvrdy(cctx)

    def recv_service(self, msg: RecvMessage):
        logging.debug(f"APP serve message {msg}")
        cctx = self.ctable.get_cctx_from_stream(msg.sid)
        if cctx is None:
            logging.warning(f'不存在与{msg.sid}关联的会话')
            return
        cctx.update_active()
        with cctx.lock:
            # de_data = InfoDTO.InfoDTOSerializer.deserialize(msg.data)
            de_data = InfoDTO.InfoDTOSerializer.deserialize_from_str(msg.data)
            if de_data is None:
                return
            self.ctable.add_data(de_data)

    def recvend_service(self, msg: RecvEndMessage):
        logging.debug(f"APP serve message {msg}")
        cctx = self.ctable.get_cctx_from_stream(msg.sid)
        if cctx is None:
            logging.warning(f'不存在与{msg.sid}关联的会话')
            return
        with cctx.lock:
            if cctx.stream_state == CSContextCoteeState.PENDING:
                server_logic_error(f"收到RECVEND, 会话context: {cctx.cid} 还未收到RECVRDY")
            elif cctx.stream_state == CSContextCoteeState.WAITRDY:
                server_logic_error(f"收到RECVEND, 会话context: {cctx.cid} 还未收到RECVRDY")
            elif cctx.stream_state == CSContextCoteeState.RECVRDY:
                self.stream_to_end(cctx)
            elif cctx.stream_state == CSContextCoteeState.RECVEND:
                logging.debug("收到RECVEND, 会话context: {cctx.cid} 流接收结束")

    def disconnect(self, id):
        subed_cctx = self.ctable.get_subscribed_by_id(id)['cctx']
        subing_cctx = self.ctable.get_subscribing_by_id(id)
        if subed_cctx is not None:
            with subed_cctx.lock:
                self.notify_send(subed_cctx, NotifyAct.FIN)
                self.cctx_to_closed(subed_cctx)

        if subing_cctx is not None:
            with subing_cctx.lock:
                self.subscribe_send(subing_cctx, SubscribeAct.FIN)
                self.cctx_to_closed(subing_cctx)