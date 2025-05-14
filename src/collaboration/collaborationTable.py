from __future__ import annotations

import threading
from typing import Dict, Iterable, Optional, Tuple
import AppType
from cachetools import LRUCache, TTLCache
from collaboration.collaborationContext import CContext
from collaboration.broadcastCollaborationContext import BCContext
from collaboration.collaborationConfig import CollaborationConfig
from config import AppConfig
from utils.InfoDTO import InfoDTO
from utils.common import server_assert


# TODO 加锁
class CollaborationTable:
    def __init__(self, cfg: AppConfig) -> None:
        
        self.cfg = cfg
        
        self.cctx_lock = threading.Lock()
        # (cid, cotor, cotee) -> cctx
        self.cctx: Dict[Tuple[AppType.cid_t, AppType.id_t, AppType.id_t], CContext] = dict()
        
        self.bcctx_lock = threading.Lock()
        # cid -> bcctx
        self.bcctx: Dict[AppType.cid_t, BCContext] = dict()

        self.stream_lock = threading.Lock()
        # sid -> cctx
        self.stream: Dict[AppType.sid_t, CContext] = dict()

        self.sendnty_lock = threading.Lock()
        # cotee id -> cctx 正在sendnty状态的cctx, 因为只有当前是cotor的时候状态才可能为sendnty，用cotee id做为索引
        self.sendnty: Dict[AppType.id_t, CContext] = dict()

        self.subscribed_lock = threading.Lock()
        # cotee id -> cctx 正在被订阅的cctx
        self.subscribed: Dict[AppType.id_t, CContext] = dict()

        self.waitnty_lock = threading.Lock()
        # cotor id -> cctx 正在waitnty状态的cctx, 因为只有当前是cotee的时候状态才可能为waitnty，用cotor id做为索引
        self.waitnty: Dict[AppType.id_t, CContext] = dict()

        self.subscribing_lock = threading.Lock()
        # cotor id -> cctx 正在订阅的cctx
        self.subscribing: Dict[AppType.id_t, CContext] = dict()

        self.data_cache_lock = threading.Lock()
        self.data_cache: TTLCache[AppType.id_t, InfoDTO] = TTLCache(self.cfg.data_cache_size, cfg.other_data_cache_ttl)  # 他车数据的缓存

    def add_cctx(self, cctx: CContext):
        with self.cctx_lock:
            self.cctx[(cctx.cid, cctx.cotor, cctx.cotee)] = cctx

    def check_cctx_exist(self, cid, cotor, cotee):
        return self.get_cctx(cid, cotor, cotee) is not None

    def get_cctx(self, cid, cotor, cotee) -> Optional[CContext]:
        with self.cctx_lock:
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
        with self.cctx_lock:
            ctable = self.cctx
            t = (cid, cotor, cotee)
            if t in ctable:
                cctx = ctable[t]
            else:
                assert False
            return cctx

    def rem_cctx(self, cctx: CContext):
        with self.cctx_lock:
            self.cctx.pop((cctx.cid, cctx.cotor, cctx.cotee))

    def get_cctx_from_stream(self, sid: AppType.sid_t) -> Optional[CContext]:
        with self.stream_lock:
            stream_table = self.stream
            if sid in stream_table:
                return stream_table[sid]
            else:
                return None

    def add_stream(self, sid, cctx):
        with self.stream_lock:
            self.stream[sid] = cctx

    def rem_stream(self, sid):
        with self.stream_lock:
            self.stream.pop(sid)

    # ============== bcctx ====================#
    def add_bcctx(self, bcctx: BCContext):
        with self.bcctx_lock:
            self.bcctx[bcctx.cid] = bcctx

    def get_bcctx(self, cid: AppType.cid_t) -> Optional[BCContext]:
        with self.bcctx_lock:
            if cid in self.bcctx:
                return self.bcctx[cid]
            else:
                return None

    def rem_bcctx(self, bcctx: BCContext):
        with self.bcctx_lock:
            self.bcctx.pop(bcctx.cid)

    # ============== cotee state table ====================#
    def add_waitnty(self, cctx: CContext):
        with self.waitnty_lock:
            self.waitnty[cctx.cotor] = cctx

    def rem_waitnty(self, cctx: CContext):
        with self.waitnty_lock:
            server_assert(cctx.cotor in self.waitnty)
            self.waitnty.pop(cctx.cotor)

    def get_waitnty_by_id(self, cotor_id):
        with self.waitnty_lock:
            return self.waitnty[cotor_id] if cotor_id in self.waitnty else None

    def add_subscribing(self, cctx: CContext):
        with self.subscribing_lock:
            self.subscribing[cctx.cotor] = cctx

    def rem_subscribing(self, cctx: CContext):
        with self.subscribing_lock:
            server_assert(cctx.cotor in self.waitnty)
            self.subscribing.pop(cctx.cotor)

    def get_subscribing_by_id(self, cotor_id) -> Optional[CContext]:
        with self.subscribing_lock:
            return self.subscribing[cotor_id] if cotor_id in self.subscribing else None

    def get_subscribing(self) -> Iterable[CContext]:
        with self.subscribing_lock:
            return self.subscribing.values()

    # ============== cotor state table ====================#
    def add_sendnty(self, cctx: CContext):
        with self.sendnty_lock:
            self.sendnty[cctx.cotee] = cctx

    def rem_sendnty(self, cctx: CContext):
        with self.sendnty_lock:
            server_assert(cctx.cotee in self.sendnty)
            self.sendnty.pop(cctx.cotee)

    def get_sendnty_by_id(self, cotee_id):
        with self.sendnty_lock:
            return self.sendnty[cotee_id] if cotee_id in self.sendnty else None

    def add_subscribed(self, cctx: CContext):
        with self.subscribed_lock:
            self.subscribed[cctx.cotee] = cctx

    def rem_subscribed(self, cctx: CContext):
        with self.subscribed_lock:
            server_assert(cctx.cotee in self.subscribed)
            self.subscribed.pop(cctx.cotee)

    def get_subscribed(self) -> Iterable[CContext]:
        with self.subscribed_lock:
            return self.subscribed.values()
    
    def get_subscribed_by_id(self, cotor_id):
        with self.subscribed_lock:
            return self.subscribing[cotor_id] if cotor_id in self.subscribing else None
