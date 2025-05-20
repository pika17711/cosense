from __future__ import annotations

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
import numpy as np
from utils import InfoDTO
from utils.common import read_binary_file, server_assert, server_logic_error, server_not_implemented

class CollaborationService():
    def __init__(self, 
                 cfg: AppConfig,
                 ctable: 'CollaborationTable',
                 perception_client: 'PerceptionRPCClient',
                 tx_handler: 'transactionHandler'
                 ) -> None:
        self.cfg = cfg 
        self.cid_gen = ContextGenerator(cfg)
        self.ctable = ctable
        self.perception_client = perception_client
        self.tx_handler = tx_handler

    def cctx_to_waitnty(self, cctx: CContext):
        server_assert(cctx.is_cotee(), "上下文角色必须是被协作者")
        server_assert(cctx.state == CContextCoteeState.PENDING)
        logging.debug(f"context: {cctx.cid}, 状态 {cctx.state} -> CContextCoteeState.WAITNTY")
        cctx.update_active()
        cctx.state = CContextCoteeState.WAITNTY
        self.ctable.add_waitnty(cctx)

    def cctx_to_subscribing(self, cctx: CContext):
        server_assert(cctx.is_cotee(), "上下文角色必须是被协作者")
        server_assert(cctx.state == CContextCoteeState.WAITNTY)
        logging.debug(f"context: {cctx.cid}, 状态 {cctx.state} -> CContextCoteeState.SUBSCRIBING")
        cctx.update_active()
        self.ctable.rem_waitnty(cctx)
        logging.debug(f"订阅 {cctx.remote_id()}, context: {cctx.cid}")
        cctx.state = CContextCoteeState.SUBSCRIBING
        self.ctable.add_subscribing(cctx)

    def cctx_to_sendnty(self, cctx: CContext):
        server_assert(cctx.is_cotor(), "上下文角色必须是协作者")
        server_assert(cctx.state == CContextCotorState.PENDING)
        logging.debug(f"context: {cctx.cid}, 状态 {cctx.state} -> CContextCotorState.SENDNTY")
        cctx.update_active()
        cctx.state = CContextCotorState.SENDNTY
        self.ctable.add_sendnty(cctx)

    def cctx_to_subscribed(self, cctx: CContext):
        server_assert(cctx.is_cotor(), "上下文角色必须是协作者")
        server_assert(cctx.state == CContextCotorState.SENDNTY)
        logging.debug(f"context: {cctx.cid}, 状态 {cctx.state} -> CContextCotorState.SUBSCRIBED")
        cctx.update_active()
        self.ctable.rem_sendnty(cctx)
        logging.debug(f"被 { cctx.remote_id()} 订阅, context: {cctx.cid}")
        cctx.state = CContextCotorState.SUBSCRIBED
        self.ctable.add_subscribed(cctx)

    def cctx_to_closed(self, cctx: CContext):
        cctx.update_active()
        if cctx.is_cotor():
            logging.debug(f"context: {cctx.cid}, 状态 {cctx.state} -> CContextCotorState.CLOSED")
            if cctx.state == CContextCotorState.SUBSCRIBED:
                self.ctable.rem_subscribed(cctx)
            cctx.state = CContextCotorState.CLOSED
        else:
            logging.debug(f"context: {cctx.cid}, 状态 {cctx.state} -> CContextCoteeState.CLOSED")            
            if cctx.state == CContextCoteeState.SUBSCRIBING:
                self.ctable.rem_subscribing(cctx)
            cctx.state = CContextCoteeState.CLOSED
        bcctx = self.ctable.get_bcctx(cctx.cid)
        if bcctx is not None:
            self.bcctx_rem_cctx(bcctx, cctx)

        self.stream_to_end(cctx)
        self.ctable.rem_cctx(cctx)

    def stream_to_waitrdy(self, cctx: CContext):
        server_assert(cctx.is_cotee())
        cctx.update_active()
        cctx.stream_state = CSContextCoteeState.WAITRDY

    def stream_to_recvrdy(self, cctx: CContext):
        server_assert(cctx.is_cotee())
        cctx.update_active()
        cctx.stream_state = CSContextCoteeState.RECVRDY

    def stream_to_sendreq(self, cctx: CContext):
        server_assert(cctx.is_cotor())
        cctx.update_active()
        cctx.stream_state = CSContextCotorState.SENDREQ

    def stream_to_sendrdy(self, cctx: CContext):
        server_assert(cctx.is_cotor())
        cctx.update_active()
        cctx.stream_state = CSContextCotorState.SENDRDY

    def stream_to_end(self, cctx: CContext):
        if cctx.is_cotor():
            cctx.stream_state = CSContextCotorState.SENDEND
        else:
            cctx.stream_state = CSContextCoteeState.RECVEND
        if cctx.have_sid():
            self.ctable.rem_stream(cctx.sid)

    def bcctx_to_waitbnnty(self, bcctx: BCContext):
        server_assert(bcctx.state == BCContextState.PENDING, "必须发送了广播订阅后，才能等待广播通知消息")
        bcctx.state = BCContextState.WAITBNTY

    def bcctx_to_closed(self, bcctx: BCContext):
        if self.state == BCContextState.PENDING:
            logging.warning(f"广播会话 {bcctx.cid}未发送广播订阅消息即被关闭")

        self.ctable.rem_bcctx(bcctx)
        self.state = BCContextState.CLOSED
    
    def bcctx_add_cctx(self, bcctx: BCContext, cctx: CContext):
        bcctx.cctx_set.add(cctx)

    def bcctx_rem_cctx(self, bcctx: BCContext, cctx: CContext):
        bcctx.cctx_set.remove(cctx)

    def get_self_coodmap(self):
        coopmap = CoopMap(self.cfg.id, CoopMapType.DEBUG, None, None)
        return coopmap

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
        coopmap_self = self.get_self_coodmap()
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
        coopmap_self = self.get_self_coodmap()
        ratio = CoopMap.calculate_overlap_ratio(coopmap, coopmap_self)
        return ratio >= self.cfg.overlap_threshold

    def check_need_subscribe(self, msg: SubscribeMessage):
        """
            是否接受remote的订阅
                目前开启订阅的操作只有两种:
                1. broadcastpub
                2. broadcastsub
                这两种都已经检查过了重叠率, 所以无需再检查
        """
        return True

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
        coopmap_self = self.get_self_coodmap()
        ratio = CoopMap.calculate_overlap_ratio(coopmap, coopmap_self)
        return ratio >= self.cfg.overlap_threshold

    def broadcastsub_send(self):
        """
            广播订阅
            描述：local开启一个新广播会话bcctx
                 启动一个新协程bcctx_loop(bcctx)，进行消息接收和处理
        """
        cid = self.cid_gen()
        coopmap = self.get_self_coodmap()
        decoopmap = CoopMap.serialize(coopmap)
        coopMapType = 1
        bearCap = 1
        bcctx = BCContext(self.cfg, cid)
        self.ctable.add_bcctx(bcctx)
        self.tx_handler.brocastsub(self.cfg.id, self.cfg.topic, cid, decoopmap, coopMapType, bearCap)
        self.bcctx_to_waitbnnty(bcctx)

    def broadcastpub_send(self):
        """
            广播推送
            描述：只需发送广播推送
        """
        coopmap = self.get_self_coodmap()
        decoopmap = CoopMap.serialize(coopmap)
        coopMapType = 1
        self.tx_handler.brocastpub(self.cfg.id, self.cfg.topic, decoopmap, coopMapType)

    def subscribe_send(self, did: Union[List[appType.id_t], appType.id_t], act:SubscribeAct=SubscribeAct.ACKUPD):
        """
            订阅
            描述：
                1. act = SubscribeAct.ACKUPD
                    订阅请求
                    local开启一个新会话cctx
                    启动一个新协程cctx_loop(cctx)，进行消息接收和处理
                2. act = SubscribeAct.FIN
                    订阅关闭
                    local是cotee，remote是cotor，不然就是代码逻辑错误

            参数：
                1. did
                    目标id列表或目标id
                2. act
                    订阅请求或订阅关闭
        """
        if type(did) is appType.id_t:
            did = [did]
        for didi in did:
            if act == SubscribeAct.ACKUPD:
                cid = self.cid_gen()
                cctx = CContext(self.cfg, cid, didi, self.cfg.id)
                self.ctable.add_cctx(cctx)
                coopmap = self.get_self_coodmap()
                decoopmap = CoopMap.serialize(coopmap)
                coopMapType = 1
                bearCap = 1
                self.tx_handler.subscribe(self.cfg.id, [didi], self.cfg.topic, act, cid, decoopmap, coopMapType, bearCap)
                self.cctx_to_waitnty(cctx)
            else:
                cctx = self.ctable.get_subscribing_by_id(didi)
                if cctx is None:
                    logging.warning(f"试图关闭不存在的订阅关系 {cctx}")
                    return
                fakecoopmap = np.array([1]).tobytes()
                coopMapType = 1
                bearCap = 0
                self.tx_handler.subscribe(self.cfg.id, [didi], self.cfg.topic, act, cctx.cid, fakecoopmap, coopMapType, bearCap)  # TODO 关闭订阅不需要传协作图
                self.cctx_to_closed(cctx)
                self.ctable.rem_cctx(cctx)

    def notify_send(self, cid, did, act=NotifyAct.ACK):
        """
            通知
            描述：
                1. act = NotifyAct.ACK
                    确认订购
                    对话cctx在收到订阅消息时新建
                    找到对应cctx，发送确认订购后，改变cctx状态
                2. act = NotifyAct.NTY
                    对话内通知
                    未实现，没理解作用
                2. act = NotifyAct.FIN
                    取消订购
                    对话cctx在收到订阅消息时新建
                    找到对应cctx，发送取消订购后，改变cctx状态
            参数：
                1. did
                2. act
        """
        self.ctable.get_cctx_or_panic(cid, self.cfg.id, did)
        if act == NotifyAct.ACK:
            coopmap = self.get_self_coodmap()
            decoopmap = CoopMap.serialize(coopmap)
            coopMapType = 1
            bearCap = 1
            self.tx_handler.notify(self.cfg.id, did, self.cfg.topic, act, cid, decoopmap, coopMapType, bearCap)
        elif act == NotifyAct.NTY:
            pass
        elif act == NotifyAct.FIN:
            fakecoopMap = np.array([1]).tobytes()
            coopMapType = 1
            bearCap = 1 
            self.tx_handler.notify(self.cfg.id, did, self.cfg.topic, act, cid, fakecoopMap, coopMapType, bearCap) # TODO 这里不需要传协作图

    def sendreq_send(self, did: appType.id_t, cid: appType.cid_t, rl, pt, aoi, mode):
        self.tx_handler.sendreq(did, cid, rl, pt, aoi, mode)

    def send_send(self, sid: appType.sid_t, data: bytes):
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
            self.sendreq_send(cctx.remote_id(), cctx.cid, rl, pt, aoi, mode)
            self.stream_to_sendreq(cctx)
            server_assert(cctx.stream_state == CSContextCotorState.SENDRDY)

    def send_data(self, cctx: CContext, data: bytes):
        if cctx.stream_state == CSContextCotorState.SENDEND:
            logging.debug(f"context: {cctx.cid} 发送结束, 发送数据失败")
            return

        if not cctx.have_sid():
            if cctx.stream_state == CSContextCotorState.SENDREQ:
                logging.debug(f"context: {cctx.cid} 获取stream中, 发送数据失败")
            elif cctx.stream_state == CSContextCotorState.PENDING:
                self.get_stream(cctx)
        else:
            self.send_send(cctx.sid, data)

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
            cid = self.cid_gen()
            cctx = CContext(self.cfg, cid, msg.oid, self.cfg.id)
            self.ctable.add_cctx(cctx)
            self.subscribe_send(msg.oid)
            self.cctx_to_waitnty(cctx)
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
            coopmap = self.get_self_coodmap()
            decoopmap = CoopMap.serialize(coopmap)
            coopMapType = 1
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
                cctx = CContext(self.cfg, bcctx.cid, msg.oid, self.cfg.id)
                self.ctable.add_cctx(cctx)
                self.bcctx_add_cctx(bcctx, cctx)
                self.subscribe_send(msg.oid, SubscribeAct.ACKUPD)
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

    def subscribe_ackupd_service(self, msg: SubscribeMessage):
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
        self.cctx_to_sendnty(cctx)
        if self.check_need_subscribe(msg):
            self.ctable.add_cctx(cctx)
            self.notify_send(cctx.cid, msg.oid)
            self.cctx_to_subscribed(cctx)
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
        if cctx.state == CContextCotorState.PENDING:
            # TODO 输出警告 代码错误 或消息错误
            logging.warning(f"收到SUBSCRIBE FIN, 但对应context:{msg.context}还未发送NOTIFY")
        elif cctx.state == CContextCotorState.SENDNTY:
            self.cctx_to_closed(cctx)
        elif cctx.state == CContextCotorState.SUBSCRIBED:
            self.cctx_to_closed(cctx)
        elif cctx.state == CContextCotorState.CLOSED:
            pass

    def subscribe_service(self, msg: SubscribeMessage):
        logging.debug(f"APP serve message {msg}")
        if msg.act == SubscribeAct.ACKUPD:
            self.subscribe_ackupd_service(msg)
        elif msg.act == SubscribeAct.FIN:
            self.subscribe_fin_service(msg)
        else:
            assert False

    def recvfile_service(self, msg: RecvFileMessage):
        logging.debug(f"APP serve message {msg}")
        cctx = self.ctable.get_cctx_or_panic(msg.context, self.cfg.id, msg.oid)
        assert cctx.state == CContextCoteeState.SUBSCRIBING
        data = read_binary_file(msg.file)
        de_data = InfoDTO.InfoDTOSerializer.deserialize(data)
        if de_data is None:
            return
        self.ctable.add_data(de_data)

    def sendfin_service(self, msg: SendFinMessage):
        logging.debug(f"APP serve message {msg}")

    def sendrdy_service(self, msg: SendRdyMessage):
        logging.debug(f"APP serve message {msg}")
        cctx = self.ctable.get_cctx_or_panic(msg.context, self.cfg.id, msg.did)
        server_assert(not cctx.have_sid())
        cctx.sid = msg.sid
        cctx.stream_state = CSContextCotorState.SENDRDY
        self.ctable.add_stream(msg.sid, cctx)
        cctx.sid_set_event.set()
    
    def recvrdy_service(self, msg: RecvRdyMessage):
        logging.debug(f"APP serve message {msg}")
        cctx = self.ctable.get_cctx_or_panic(msg.context, self.cfg.id, msg.oid)
        server_assert(not cctx.have_sid())
        cctx.sid = msg.sid
        self.stream_to_recvrdy(cctx)
        cctx.sid_set_event.set()

    def recv_service(self, msg: RecvMessage):
        logging.debug(f"APP serve message {msg}")
        cctx = self.ctable.get_cctx_from_stream(msg.sid)
        if cctx is None:
            logging.warning(f'不存在与{msg.sid}关联的会话')
            return
        de_data = InfoDTO.InfoDTOSerializer.deserialize(msg.data)
        if de_data is None:
            return
        self.ctable.add_data(de_data)

    def recvend_service(self, msg: RecvEndMessage):
        logging.debug(f"APP serve message {msg}")
        cctx = self.ctable.get_cctx_from_stream(msg.sid)
        if cctx is None:
            logging.warning(f'不存在与{msg.sid}关联的会话')
            return
        if cctx.stream_state == CSContextCoteeState.PENDING:
            server_logic_error(f"收到RECVEND, 会话context: {cctx.cid} 还未收到RECVRDY")
        elif cctx.stream_state == CSContextCoteeState.WAITRDY:
            server_logic_error(f"收到RECVEND, 会话context: {cctx.cid} 还未收到RECVRDY")
        elif cctx.stream_state == CSContextCoteeState.RECVRDY:
            self.stream_to_end(cctx)
        elif cctx.stream_state == CSContextCoteeState.RECVEND:
            logging.debug("收到RECVEND, 会话context: {cctx.cid} 流接收结束")

    def disconnect(self, id):
        subed_cctx = self.ctable.get_subscribed_by_id(id)
        subing_cctx = self.ctable.get_subscribing_by_id(id)
        if subed_cctx is not None:
            self.notify_send(subed_cctx.cid, subed_cctx.remote_id(), NotifyAct.FIN)

        if subing_cctx is not None:
            self.subscribe_send(subing_cctx.cid, subing_cctx.remote_id(), SubscribeAct.FIN)