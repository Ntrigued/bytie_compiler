from abc import ABC, abstractmethod
from bytie.ast import Program


class ASTPass(ABC):
    @abstractmethod
    def run(self, ast: Program) -> Program:
        pass
