
import random
from typing import Optional
from appConfig import AppConfig
import appType
from utils.common import mstime, string_to_32_hex

class ContextGenerator:
    def __init__(self, cfg: AppConfig, seed: Optional[int]=None) -> None:
        self.id = cfg.id
        if seed:
            self.counter = seed
        else:
            self.counter = random.randint(1, 1 << 20)
    
    def cid_gen(self) -> appType.cid_t:
        self.counter += 1
        return string_to_32_hex(str(self.id) + str(mstime()) + str(self.counter))

    def __call__(self) -> appType.cid_t:
        return self.cid_gen()