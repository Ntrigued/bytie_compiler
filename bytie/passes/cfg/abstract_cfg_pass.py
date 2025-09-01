from abc import ABC, abstractmethod
from bytie.cfg import CFG


class AbstractCFGPass(ABC):
    @abstractmethod
    def run(self, cfg: CFG) -> CFG:
        pass