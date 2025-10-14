from dataclasses import dataclass
from typing import Optional


@dataclass
class BearInfo:
    ipv4: Optional[str]
    ipv6: Optional[str]
    port: Optional[str]
    sid: int = 0
    rl: int = 1
    netType: int = 0
    netPto: int = 0
    transPto: int = 0

    def to_dict(self):
        return self.__dict__

