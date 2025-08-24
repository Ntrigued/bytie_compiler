"""Type definitions and helpers for Bytie.

This module defines the runtime type system used by the Bytie interpreter.
It includes classes for representing type specifications and values, as well
as utilities for checking values against type specifications and performing
numeric conversions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import math


@dataclass(frozen=True)
class TypeSpec:
    """Represents a Bytie type specification.

    A type is described by its `kind` (one of 'Integer', 'Double', 'Str',
    'Array', 'Map', 'Error', or 'None') and optionally type arguments for
    generics. For example, `Array<Integer>` becomes
    `TypeSpec(kind='Array', args=(TypeSpec(kind='Integer'),))`.
    """
    kind: str
    args: Tuple['TypeSpec', ...] = ()

    def __repr__(self) -> str:
        if not self.args:
            return self.kind
        inner = ", ".join(repr(a) for a in self.args)
        return f"{self.kind}<{inner}>"

    # Convenience constructors
    @staticmethod
    def integer() -> 'TypeSpec':
        return TypeSpec('Integer')

    @staticmethod
    def double() -> 'TypeSpec':
        return TypeSpec('Double')

    @staticmethod
    def string() -> 'TypeSpec':
        return TypeSpec('Str')

    @staticmethod
    def array(elem: 'TypeSpec') -> 'TypeSpec':
        return TypeSpec('Array', (elem,))

    @staticmethod
    def map(value: 'TypeSpec') -> 'TypeSpec':
        return TypeSpec('Map', (value,))

    @staticmethod
    def error() -> 'TypeSpec':
        return TypeSpec('Error')

    @staticmethod
    def none() -> 'TypeSpec':
        return TypeSpec('None')


class NoneVal:
    """Marker object for the Bytie `None` value."""
    def __repr__(self) -> str:
        return 'None'


@dataclass
class ErrorVal:
    """Represents a Bytie error value.

    Errors are first-class values in Bytie. They carry a name and a
    message. They are truthy for the purposes of boolean operators but
    comparisons involving errors are not defined.
    """
    name: str
    message: str

    def __repr__(self) -> str:
        return f"Error(name={self.name!r}, message={self.message!r})"


@dataclass
class ArrayVal:
    """Represents a Bytie array value.

    An array has a declared element type (a TypeSpec) and a list of
    contained items. The interpreter enforces that all items conform to
    the declared element type when the array is created and when items are
    assigned.
    """
    elem_type: TypeSpec
    items: List[Any]

    def __repr__(self) -> str:
        return f"Array({self.elem_type!r}, {self.items!r})"


@dataclass
class MapVal:
    """Represents a Bytie map value.

    Maps have string keys and values of a declared type. All inserted
    values must conform to that type. Missing keys raise KeyError when
    accessed.
    """
    value_type: TypeSpec
    entries: Dict[str, Any]

    def __repr__(self) -> str:
        return f"Map({self.value_type!r}, {self.entries!r})"


def round_to_int_away_from_zero(x: float) -> int:
    """Round a floating point number to the nearest integer away from zero.

    Python's built-in round uses bankers rounding, so we implement the
    custom rule required by Bytie: halves are rounded away from zero.
    """
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def check_value(value: Any, spec: TypeSpec) -> bool:
    """Check whether a runtime value matches a type specification.

    This function returns True if the value conforms to the given type
    specification. If the value does not match the type, it raises a
    TypeError (not a Bytie error) with a descriptive message. The caller
    should catch such errors and raise Bytie errors as appropriate.
    """
    kind = spec.kind
    if kind == 'Integer':
        if isinstance(value, bool):  # bool is a subclass of int; treat separately
            return True
        if isinstance(value, int):
            return True
        raise TypeError(f"expected Integer, got {type(value).__name__}")
    elif kind == 'Double':
        if isinstance(value, float):
            return True
        raise TypeError(f"expected Double, got {type(value).__name__}")
    elif kind == 'Str':
        if isinstance(value, str):
            return True
        raise TypeError(f"expected Str, got {type(value).__name__}")
    elif kind == 'Error':
        if isinstance(value, ErrorVal):
            return True
        raise TypeError(f"expected Error, got {type(value).__name__}")
    elif kind == 'None':
        if isinstance(value, NoneVal):
            return True
        raise TypeError(f"expected None, got {type(value).__name__}")
    elif kind == 'Array':
        # spec.args should have length 1
        if not isinstance(value, ArrayVal):
            raise TypeError(f"expected Array, got {type(value).__name__}")
        # check element type matches spec.args[0]
        # If no type argument is provided (spec.args empty), accept any Array
        if len(spec.args) == 0:
            return True
        elem_type = spec.args[0]
        # If the specified element type itself has no args (e.g., Array<>) treat as a wildcard
        # that matches any array element type.  This allows Array<Array> to accept
        # arrays with any element type, matching the examples.
        if elem_type.kind == 'Array' and len(elem_type.args) == 0:
            return True
        # otherwise require exact match on element type
        if value.elem_type != elem_type:
            raise TypeError(f"array element type mismatch: expected {elem_type}, got {value.elem_type}")
        # Optionally check each element recursively here; trust creation to enforce
        return True
    elif kind == 'Map':
        if not isinstance(value, MapVal):
            raise TypeError(f"expected Map, got {type(value).__name__}")
        val_type = spec.args[0]
        if value.value_type != val_type:
            raise TypeError(f"map value type mismatch: expected {val_type}, got {value.value_type}")
        return True
    else:
        raise TypeError(f"unknown type spec: {spec}")


def type_name(value: Any) -> str:
    """Return the Bytie type name of a runtime value."""
    if isinstance(value, bool):
        # Booleans behave like integers
        return 'Integer'
    if isinstance(value, int):
        return 'Integer'
    if isinstance(value, float):
        return 'Double'
    if isinstance(value, str):
        return 'Str'
    if isinstance(value, ArrayVal):
        return f"Array<{repr(value.elem_type)}>"
    if isinstance(value, MapVal):
        return f"Map<{repr(value.value_type)}>"
    if isinstance(value, ErrorVal):
        return 'Error'
    if isinstance(value, NoneVal):
        return 'None'
    return type(value).__name__


def to_string(value: Any) -> str:
    """Convert a Bytie value to its string representation for printing.

    This differs from conversion in that arrays and maps are shown in a
    debug-friendly format rather than being converted to user-visible
    strings. Use `convert_value` for strict type conversion.
    """
    if isinstance(value, bool):
        # booleans behave like integers
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # Use repr for a concise round-trip representation
        return repr(value)
    if isinstance(value, str):
        return value
    if isinstance(value, ArrayVal):
        # Show a simple preview; we don't print full nested structures
        return '[' + ', '.join(to_string(item) for item in value.items) + ']'
    if isinstance(value, MapVal):
        entries = ', '.join(f"{k}: {to_string(v)}" for k, v in value.entries.items())
        return '{' + entries + '}'
    if isinstance(value, ErrorVal):
        return f"<Error {value.name}: {value.message}>"
    if isinstance(value, NoneVal):
        return 'None'
    return str(value)


def convert_value(target: TypeSpec, source: TypeSpec, value: Any) -> Any:
    """Convert a value from one type to another according to Bytie rules.

    Raises a ValueError or TypeError if the conversion is not allowed or
    fails. Conversion rules are documented in the specification. The
    `source` parameter is not currently used but is included for
    completeness and future extension.
    """
    # Disallow Error conversions
    if target.kind == 'Error' or source.kind == 'Error' or isinstance(value, ErrorVal):
        raise TypeError('cannot convert Error type')
    # Disallow Map conversions
    if target.kind == 'Map' or source.kind == 'Map' or isinstance(value, MapVal):
        raise TypeError('cannot convert Map type')
    # Disallow Array conversions except Array<Str> <-> Str
    if target.kind == 'Array' and source.kind != 'Array':
        # converting from non-array to array
        if target.args[0].kind == 'Str' and isinstance(value, str):
            # Str -> Array<Str>
            return ArrayVal(target.args[0], [ch for ch in value])
        else:
            raise TypeError(f'cannot convert from {source} to {target}')
    if source.kind == 'Array' and target.kind != 'Array':
        if source.args[0].kind == 'Str' and isinstance(value, ArrayVal) and value.elem_type.kind == 'Str':
            # Array<Str> -> Str
            return ''.join(value.items)
        else:
            raise TypeError(f'cannot convert from {source} to {target}')
    # Numeric conversions
    if target.kind == 'Integer' and isinstance(value, float):
        return round_to_int_away_from_zero(value)
    if target.kind == 'Integer' and isinstance(value, int):
        return value
    if target.kind == 'Integer' and isinstance(value, str):
        # parse string to integer
        try:
            return int(value)
        except Exception:
            raise ValueError(f'cannot parse int from {value!r}')
    if target.kind == 'Double' and isinstance(value, int):
        return float(value)
    if target.kind == 'Double' and isinstance(value, float):
        return value
    if target.kind == 'Double' and isinstance(value, str):
        try:
            return float(value)
        except Exception:
            raise ValueError(f'cannot parse double from {value!r}')
    # Str conversions
    if target.kind == 'Str':
        # convert any to string via to_string for conversion
        return to_string(value)
    # Identity conversions
    if type_name(value) == target.kind:
        return value
    raise TypeError(f'cannot convert from {source} to {target}')