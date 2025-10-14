from __future__ import annotations

import copy
import logging
import threading
from typing import Dict, Iterable, List, Optional, Tuple
import appType
from cachetools import TTLCache
from collaboration.collaborationContext import CContext
from collaboration.broadcastCollaborationContext import BCContext
from appConfig import AppConfig
from collaboration.coopMap import CoopMap
from utils.InfoDTO import InfoDTO
from utils.common import server_assert


class CollaborationTable:
    """
        记录协作的状态
            cctx=CollaborationContext
            cid=contextid
            bcctx=BroadcastCollaborationContext
            cotee=被协作者，订阅者，接收来自协作者的数据
            cotor=协作者，被订阅者，向被协作者发送数据
    """
    def __init__(self, cfg: AppConfig) -> None: 
        self.cfg = cfg
        
        self.cctx_lock = threading.Lock()
        # (cid, cotor, cotee) -> cctx
        # 由broadcastsub开启的协作对话使用与广播对话相同的cid，同一个cid可能对应不同的remote，所以不能使用单独的cid做key
        self.cctx: Dict[Tuple[appType.cid_t, appType.id_t, appType.id_t], CContext] = dict()
        
        self.bcctx_lock = threading.Lock()
        # cid -> bcctx
        self.bcctx: Dict[appType.cid_t, BCContext] = dict()

        self.stream_lock = threading.Lock()
        # sid -> cctx
        self.stream: Dict[appType.sid_t, CContext] = dict()

        self.sendnty_lock = threading.Lock()
        # cotee id -> cctx 正在sendnty状态的cctx, 因为只有当前是cotor的时候状态才可能为sendnty，用cotee id做为索引
        self.sendnty: Dict[appType.id_t, CContext] = dict()

        self.subscribed_lock = threading.Lock()
        # cotee id -> cctx 正在被订阅的cctx
        self.subscribed: Dict[appType.id_t, CContext] = dict()

        self.waitnty_lock = threading.Lock()
        # cotor id -> cctx 正在waitnty状态的cctx, 因为只有当前是cotee的时候状态才可能为waitnty，用cotor id做为索引
        self.waitnty: Dict[appType.id_t, CContext] = dict()

        self.subscribing_lock = threading.Lock()
        # cotor id -> cctx 正在订阅的cctx
        self.subscribing: Dict[appType.id_t, CContext] = dict()

        self.data_cache_lock = threading.Lock()
        # id -> data 缓存的他车数据
        self.data_cache: TTLCache[appType.id_t, InfoDTO] = TTLCache(self.cfg.other_data_cache_size, cfg.other_data_cache_ttl)  # 他车数据的缓存

        # id -> coopmap 缓存的他车协作图
        self.coopmap_cache_lock = threading.Lock()
        self.coopmap_cache: TTLCache[appType.id_t, CoopMap] = TTLCache(self.cfg.other_data_cache_size, cfg.other_data_cache_ttl)

    def add_cctx(self, cctx: CContext):
        with self.cctx_lock:
            logging.debug(f"新增CContext: {cctx}")
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

    def get_all_cctx(self) -> List[CContext]:
        with self.cctx_lock:
            return list(self.cctx.values())

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
            t = (cctx.cid, cctx.cotor, cctx.cotee)
            if t in self.cctx:
                self.cctx.pop((cctx.cid, cctx.cotor, cctx.cotee))
            else:
                logging.warning(f'删除不存在的cctx {cctx}')

    def get_cctx_from_stream(self, sid: appType.sid_t) -> Optional[CContext]:
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

    def get_bcctx(self, cid: appType.cid_t) -> Optional[BCContext]:
        with self.bcctx_lock:
            if cid in self.bcctx:
                return self.bcctx[cid]
            else:
                return None

    def get_all_bcctx(self) -> List[BCContext]:
        with self.bcctx_lock:
            return list(self.bcctx.values())

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
            server_assert(cctx.cotor in self.subscribing)
            self.subscribing.pop(cctx.cotor)

    def get_subscribing_by_id(self, cotor_id) -> Optional[CContext]:
        with self.subscribing_lock:
            return self.subscribing[cotor_id] if cotor_id in self.subscribing else None

    def get_subscribing(self) -> List[CContext]:
        with self.subscribing_lock:
            return list(self.subscribing.values())

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

    def get_subscribed(self) -> List[CContext]:
        with self.subscribed_lock:
            return list(self.subscribed.values())

    def pop_subscribed(self) -> List[CContext]:
        with self.subscribed_lock:
            return list(self.subscribed.values())
    
    def get_subscribed_by_id(self, cotor_id):
        with self.subscribed_lock:
            return self.subscribed[cotor_id] if cotor_id in self.subscribed else None

    def add_data(self, data: InfoDTO):
        with self.data_cache_lock:
            self.data_cache[data.id] = data

    def get_all_data(self):
        with self.data_cache_lock:
            datas_copy = [copy.deepcopy(info) for info in self.data_cache.values()]
        return datas_copy

    def pop_all_data(self):
        with self.data_cache_lock:
            datas_copy = [copy.deepcopy(info) for info in self.data_cache.values()]
            self.data_cache.clear()
        return datas_copy

    def add_coopmap(self, oid, coopMap: CoopMap):
        with self.coopmap_cache_lock:
            self.coopmap_cache[oid] = coopMap

    def update_coopmap(self, oid, coopMap: CoopMap):
        with self.coopmap_cache_lock:
            self.coopmap_cache[oid] = coopMap

    def get_coopmap(self, oid):
        with self.coopmap_cache_lock:
            return self.coopmap_cache.get(oid)

    def pop_coopmap(self, oid):
        with self.coopmap_cache_lock:
            return self.coopmap_cache.pop(oid) if oid in self.coopmap_cache else None
