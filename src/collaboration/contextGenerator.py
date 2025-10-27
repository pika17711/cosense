
import random
from typing import Optional
from appConfig import AppConfig
import appType
from utils.common import mstime, string_to_32_hex

class ContextGenerator:
    """
        context id 生成器

        根据通信模块的要求，cid需要是32位16进制数字，于是用md5实现
    """

    def __init__(self, cfg: AppConfig, seed: Optional[int]=None) -> None:
        self.id = cfg.id
        if seed:
            self.counter = seed
        else:
            self.counter = random.randint(1, 1 << 20)
    
    # def cid_gen(self) -> appType.cid_t:
    #     self.counter += 1
    #     return string_to_32_hex(str(self.id) + str(mstime()) + str(self.counter))

    def cid_gen(self) -> appType.cid_t:
        self.counter += 1

        hex_digest = string_to_32_hex(str(self.id) + str(mstime()) + str(self.counter))

        cid_int = int(hex_digest[:8], 16)
        cid_int = cid_int & 0xFFFFFFFF  # 确保它是一个32位无符号整数

        return cid_int

    def __call__(self) -> appType.cid_t:
        return self.cid_gen()