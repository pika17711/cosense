from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple
import AppType
from cachetools import LRUCache
from collaboration.collaborationContext import CContext
from collaboration.broadcastCollaborationContext import BCContext
from collaboration.collaborationConfig import CollaborationConfig
from utils.common import server_assert


# TODO 加锁
class CollaborationTable:
    def __init__(self, cfg: CollaborationConfig) -> None:

        self.cfg = cfg
        # (cid, cotor, cotee) -> cctx
        self.cctx: Dict[Tuple[AppType.cid_t, AppType.id_t, AppType.id_t], CContext] = dict()
        # cid -> bcctx
        self.bcctx: Dict[AppType.cid_t, BCContext] = dict()
        # sid -> cctx
        self.stream: Dict[AppType.sid_t, CContext] = dict()

        # cotee id -> cctx 正在sendnty状态的cctx, 因为只有当前是cotor的时候状态才可能为sendnty，用cotee id做为索引
        self.sendnty: Dict[AppType.id_t, CContext] = dict()
        # cotee id -> cctx 正在被订阅的cctx
        self.subscribed: Dict[AppType.id_t, CContext] = dict()

        # cotor id -> cctx 正在waitnty状态的cctx, 因为只有当前是cotee的时候状态才可能为waitnty，用cotor id做为索引
        self.waitnty: Dict[AppType.id_t, CContext] = dict()
        # cotor id -> cctx 正在订阅的cctx
        self.subscribing: Dict[AppType.id_t, CContext] = dict()

        self.data_cache = LRUCache(self.cfg.data_cache_size)  # 他车数据的缓存

    def add_cctx(self, cctx: CContext):
        self.cctx[(cctx.cid, cctx.cotor, cctx.cotee)] = cctx

    def check_cctx_exist(self, cid, cotor, cotee):
        return self.get_cctx(cid, cotor, cotee) is not None

    def get_cctx(self, cid, cotor, cotee) -> Optional[CContext]:
        ctable = self.cctx
        t = (cid, cotor, cotee)
        if t in ctable:
            cctx = ctable[t]
        else:
            cctx = None
        return cctx

    def get_cctx_or_panic(self, cid, cotor, cotee) -> CContext:
        """
            得到cctx，若不存在，则panic
        """
        ctable = self.cctx
        t = (cid, cotor, cotee)
        if t in ctable:
            cctx = ctable[t]
        else:
            assert False
        return cctx

    def rem_cctx(self, cctx: CContext):
        self.cctx.pop((cctx.cid, cctx.cotor, cctx.cotee))

    def get_cctx_from_stream(self, sid: AppType.sid_t) -> Optional[CContext]:
        stream_table = self.stream
        if sid in stream_table:
            return stream_table[sid]
        else:
            return None

    def add_stream(self, sid, cctx):
        self.stream[sid] = cctx

    def rem_stream(self, sid):
        self.stream.pop(sid)

    # ============== bcctx ====================#
    def add_bcctx(self, bcctx: BCContext):
        self.bcctx[bcctx.cid] = bcctx

    def get_bcctx(self, cid: AppType.cid_t) -> Optional[BCContext]:
        if cid in self.bcctx:
            return self.bcctx[cid]
        else:
            return None

    def rem_bcctx(self, bcctx: BCContext):
        self.bcctx.pop(bcctx.cid)

    # ============== cotee state table ====================#
    def add_waitnty(self, cctx: CContext):
        self.waitnty[cctx.cotor] = cctx

    def rem_waitnty(self, cctx: CContext):
        server_assert(cctx.cotor in self.waitnty)
        self.waitnty.pop(cctx.cotor)

    def get_waitnty_by_id(self, cotor_id):
        return self.waitnty[cotor_id] if cotor_id in self.waitnty else None

    def add_subscribing(self, cctx: CContext):
        self.subscribing[cctx.cotor] = cctx

    def rem_subscribing(self, cctx: CContext):
        server_assert(cctx.cotor in self.waitnty)
        self.subscribing.pop(cctx.cotor)

    def get_subscribing_by_id(self, cotor_id) -> Optional[CContext]:
        return self.subscribing[cotor_id] if cotor_id in self.subscribing else None

    def get_subscribing(self) -> Iterable[CContext]:
        return self.subscribing.values()

    # ============== cotor state table ====================#
    def add_sendnty(self, cctx: CContext):
        self.sendnty[cctx.cotee] = cctx

    def rem_sendnty(self, cctx: CContext):
        server_assert(cctx.cotee in self.sendnty)
        self.sendnty.pop(cctx.cotee)

    def get_sendnty_by_id(self, cotee_id):
        return self.sendnty[cotee_id] if cotee_id in self.sendnty else None

    def add_subscribed(self, cctx: CContext):
        self.subscribed[cctx.cotee] = cctx

    def rem_subscribed(self, cctx: CContext):
        server_assert(cctx.cotee in self.subscribed)
        self.subscribed.pop(cctx.cotee)

    def get_subscribed(self) -> Iterable[CContext]:
        return self.subscribed.values()
    
    def get_subscribed_by_id(self, cotor_id):
        return self.subscribing[cotor_id] if cotor_id in self.subscribing else None