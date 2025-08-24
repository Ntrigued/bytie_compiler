from dataclasses import dataclass
from typing import Any, Optional
from bytie.types import TypeSpec

@dataclass
class BuiltinFunction:
    name: str
    arity: Optional[int]
    return_type: Optional[TypeSpec]
    fn: Any
    def __repr__(self) -> str:
        return f"<builtin {self.name}>"