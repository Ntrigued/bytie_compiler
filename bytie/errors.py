from typing import Any
from bytie.types import ErrorVal


class BytieError(Exception):
    """Exception type used to propagate Bytie runtime errors."""
    def __init__(self, err: ErrorVal):
        super().__init__(f"BytieError: {err.name}: {err.message}")
        self.err = err


class ReturnSignal(Exception):
    """Internal exception to handle return statements in functions."""
    def __init__(self, value: Any):
        super().__init__('return')
        self.value = value