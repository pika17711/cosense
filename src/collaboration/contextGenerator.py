
import random
from typing import Optional
import AppType
from utils.common import mstime, string_to_32_hex
from collaboration.collaborationConfig import CollaborationConfig

class ContextGenerator:
    def __init__(self, cfg: CollaborationConfig, seed: Optional[int]=None) -> None:
        self.id = cfg.id
        if seed:
            self.counter = seed
        else:
            self.counter = random.randint(1, 1 << 20)
    
    def cid_gen(self) -> AppType.cid_t:
        self.counter += 1
        return string_to_32_hex(str(self.id) + str(mstime()) + str(self.counter))

    def __call__(self) -> AppType.cid_t:
        return self.cid_gen()