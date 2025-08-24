from typing import Any, Dict, Optional, Set
from bytie.errors import BytieError
from bytie.types import ErrorVal, TypeSpec, ArrayVal, MapVal, NoneVal, check_value


class Environment:
    """Represents a scope environment mapping identifiers to values and metadata."""
    def __init__(self, parent: Optional['Environment'] = None):
        self.parent = parent
        self.values: Dict[str, Any] = {}
        self.consts: Dict[str, bool] = {}
        self.types: Dict[str, TypeSpec] = {}

    @property
    def export_names(self) -> Set[str]:
        # Exports: all builtins
        return set(self.values.keys())  # attribute for exports

    def get(self, name: str) -> Any:
        if name in self.values:
            return self.values[name]
        if self.parent:
            return self.parent.get(name)
        raise BytieError(ErrorVal('NameError', f'undefined variable {name}'))

    def set(self, name: str, value: Any):
        if name in self.consts and self.consts[name]:
            raise BytieError(ErrorVal('TypeError', f'cannot assign to const {name}'))
        # Determine where to set: if already declared in this env, set here, otherwise bubble to parent if exists
        if name in self.values or self.parent is None:
            # type check if declared
            if name in self.types:
                try:
                    check_value(value, self.types[name])
                except TypeError as e:
                    raise BytieError(ErrorVal('TypeError', str(e)))
            self.values[name] = value
        else:
            self.parent.set(name, value)

    def declare(self, name: str, type_spec: TypeSpec, value: Any, is_const: bool):
        if name in self.values:
            raise BytieError(ErrorVal('TypeError', f'variable {name} already declared'))
        # type check
        if value is not None:
            try:
                check_value(value, type_spec)
            except TypeError as e:
                raise BytieError(ErrorVal('TypeError', str(e)))
        else:
            # assign default value based on type
            value = self.default_value(type_spec)
        self.values[name] = value
        self.consts[name] = is_const
        self.types[name] = type_spec

    def default_value(self, type_spec: TypeSpec):
        kind = type_spec.kind
        if kind == 'Integer':
            return 0
        if kind == 'Double':
            return 0.0
        if kind == 'Str':
            return ''
        if kind == 'Array':
            return ArrayVal(type_spec.args[0], [])
        if kind == 'Map':
            return MapVal(type_spec.args[0], {})
        if kind == 'Error':
            return None  # errors must be created via error()
        if kind == 'None':
            return NoneVal()
        return None
