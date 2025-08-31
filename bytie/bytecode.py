"""
Bytecode execution engine for the Bytie language.

This module defines a very small stack‑based virtual machine and associated
data structures for executing bytecode compiled from Bytie ASTs.  The
implementation here is intentionally minimal: only the instructions
necessary to support the current language features are implemented, and
additional behaviour will be added incrementally as the lowering logic
evolves.  A BytecodeProgram is a serialisable container which stores
constant pools, import tables, global symbol names and a list of
functions.  Each function carries its own instruction list and local
variable metadata.  The BytecodeVM is responsible for linking a program
against the existing interpreter’s module loader (to ensure complete
parity with the AST interpreter) and for executing the bytecode.

When new instructions are added to the lowering pass they should be
implemented here, preserving the exact semantics of the interpreter.
All numeric operations, truthiness checks, string concatenations and
error semantics delegate to helpers found in the existing interpreter
module so that behaviour does not diverge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import json

from .enviironment import Environment
from .types import (
    TypeSpec,
    NoneVal,
    ErrorVal,
    ArrayVal,
    MapVal,
    check_value,
    convert_value,
    to_string,
    type_name,
    round_to_int_away_from_zero,
)
from .errors import BytieError
from .builtin_function import BuiltinFunction
from .interpreter import Interpreter


class Opcode:
    """Enumeration of supported bytecode instructions.

    The opcodes are stored as simple string constants.  The
    implementation of each opcode lives in BytecodeVM._dispatch().
    """

    # Stack and constants
    NOP = "NOP"
    POP = "POP"
    DUP = "DUP"
    ICONST = "ICONST"  # push integer constant by index
    DCONST = "DCONST"  # push double constant by index
    SCONST = "SCONST"  # push string constant by index
    TRUE = "TRUE"
    FALSE = "FALSE"
    NULL = "NULL"

    # Locals and globals
    LOAD_LOCAL = "LOAD_LOCAL"
    STORE_LOCAL = "STORE_LOCAL"
    LOAD_GLOBAL = "LOAD_GLOBAL"
    STORE_GLOBAL = "STORE_GLOBAL"

    # Arithmetic and logic
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"
    MOD = "MOD"
    NEG = "NEG"
    NOT = "NOT"
    EQ = "EQ"
    NE = "NE"
    LT = "LT"
    LE = "LE"
    GT = "GT"
    GE = "GE"

    # Control flow
    JUMP = "JUMP"
    JUMP_IF_FALSE = "JUMP_IF_FALSE"
    RET = "RET"

    # Functions & calls
    CALL = "CALL"
    LOAD_FUNC = "LOAD_FUNC"

    # Collections & members
    BUILD_ARRAY = "BUILD_ARRAY"
    BUILD_MAP = "BUILD_MAP"
    INDEX_GET = "INDEX_GET"
    INDEX_SET = "INDEX_SET"
    GET_FIELD = "GET_FIELD"
    SET_FIELD = "SET_FIELD"

    # Exceptions
    SETUP_TRY = "SETUP_TRY"
    POP_TRY = "POP_TRY"
    RAISE = "RAISE"

    # Utility
    TO_STR = "TO_STR"

    # Type constants
    TCONST = "TCONST"  # push TypeSpec constant by index


@dataclass
class BytecodeFunction:
    """Represents a single compiled function.

    Attributes:
        name: The name of the function (used when binding into globals).
        params: The number of positional arguments this function accepts.
        locals: Optional list of local variable names for debug.
        nlocals: Total number of local slots (including parameters).
        code: A list of instructions; each element is a list starting
              with an opcode string followed by zero or more arguments.
    """

    name: str
    params: int
    locals: List[str]
    nlocals: int
    code: List[List[Any]]


@dataclass
class BytecodeProgram:
    """Container for a complete compiled bytecode program."""

    magic: str
    version: int
    const_pool: Dict[str, List[Any]]
    imports: List[Dict[str, Any]]
    globals: List[str]
    functions: List[BytecodeFunction]
    entry: int
    # Additional metadata: constness and type specifications for globals.
    global_const: Dict[str, bool] | None = None
    global_types: Dict[str, Dict[str, Any]] | None = None

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> 'BytecodeProgram':
        """Construct a BytecodeProgram from a plain dict parsed from JSON."""
        magic = obj.get("magic")
        version = obj.get("version")
        const_pool = obj.get("const_pool", {})
        # Rehydrate TypeSpec constants into actual objects
        if "types" in const_pool:
            from .types import TypeSpec as _TypeSpec
            def _from_dict_type(d):
                # d is a dict with 'kind' and 'args'
                kind = d.get('kind')
                args = d.get('args', [])
                return _TypeSpec(kind, tuple(_from_dict_type(a) if isinstance(a, dict) else a for a in args))
            types_list = []
            for entry in const_pool["types"]:
                if isinstance(entry, dict):
                    types_list.append(_from_dict_type(entry))
                else:
                    types_list.append(entry)
            const_pool["types"] = types_list
        imports = obj.get("imports", [])
        globals_ = obj.get("globals", [])
        functions = []
        for f in obj.get("functions", []):
            functions.append(
                BytecodeFunction(
                    name=f.get("name", ""),
                    params=f.get("params", 0),
                    locals=f.get("locals", []),
                    nlocals=f.get("nlocals", 0),
                    code=f.get("code", []),
                )
            )
        entry = obj.get("entry", 0)
        return cls(
            magic=magic,
            version=version,
            const_pool=const_pool,
            imports=imports,
            globals=globals_,
            functions=functions,
            entry=entry,
            global_const=obj.get("global_const"),
            global_types=obj.get("global_types"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this program to a dict suitable for JSON encoding."""
        result = {
            "magic": self.magic,
            "version": self.version,
            "const_pool": self.const_pool,
            "imports": self.imports,
            "globals": self.globals,
            "functions": [
                {
                    "name": f.name,
                    "params": f.params,
                    "locals": f.locals,
                    "nlocals": f.nlocals,
                    "code": f.code,
                }
                for f in self.functions
            ],
            "entry": self.entry,
        }
        # If there are type constants, serialise them to dicts
        if "types" in self.const_pool:
            def _to_dict_type(ts):
                # ts is a TypeSpec
                return {
                    "kind": ts.kind,
                    "args": [_to_dict_type(a) for a in ts.args],
                }
            result_const_pool = dict(self.const_pool)
            result_const_pool["types"] = [
                _to_dict_type(ts) if not isinstance(ts, dict) else ts
                for ts in self.const_pool["types"]
            ]
            result["const_pool"] = result_const_pool
        else:
            result["const_pool"] = self.const_pool
        # global_const and global_types added later below
        if self.global_const is not None:
            result["global_const"] = self.global_const
        if self.global_types is not None:
            result["global_types"] = self.global_types
        return result


class _Frame:
    """Internal representation of a call frame."""

    def __init__(self, func_index: int, locals_size: int, return_ip: int | None):
        self.func_index = func_index
        self.locals: List[Any] = [None] * locals_size
        self.ip: int = 0
        self.return_ip = return_ip
        self.try_stack: List[Tuple[int, int]] = []  # list of (handler_ip, stack_depth)


class FunctionValue:
    """Represents a function (user defined) at runtime."""

    def __init__(self, func_index: int, param_count: int):
        self.func_index = func_index
        self.param_count = param_count

    def __repr__(self) -> str:
        return f"<function {self.func_index}>"


class BytecodeVM:
    """A simple stack machine for executing BytecodeProgram objects."""

    def __init__(self, program: BytecodeProgram, debug_level: int = 0, debug_file: str = 'debug.txt'):
        self.program = program
        self.debug_level = debug_level
        self.debug_file = debug_file
        self.debug_fp = open(debug_file, 'w') if debug_level > 0 else None
        # interpreter used for module resolution and helper functions
        self._interpreter = Interpreter(debug_level=0)  # isolate debug
        # Will be populated in link()
        self.global_env = Environment()
        # Map of global name to const and type information; copy from interpreter when linking
        self.global_consts: Dict[str, bool] = {}
        self.global_types: Dict[str, TypeSpec] = {}
        # pending const assignments: used to allow initial declaration of const vars
        self._pending_const: Dict[str, bool] = {}

    def _log(self, msg: str):
        if self.debug_level > 0 and self.debug_fp:
            self.debug_fp.write(msg + '\n')
            self.debug_fp.flush()

    def link(self) -> None:
        """Resolve imports and initialise the global environment.

        This method should be called exactly once before executing the
        program.  It preloads the standard modules via the interpreter
        to ensure parity with the AST interpreter, binds imported
        symbols into the global environment, and sets up global
        placeholders for compiled functions and variables.
        """
        # Start with a fresh environment; copy global exposures from interpreter
        # The interpreter initialises Standard and Standard_IO modules and
        # exposes error() and convert() in its global_env.  We'll copy those
        # into our VM's global environment exactly.
        self.global_env = Environment()
        std_interpreter = self._interpreter
        # Copy exposed builtins from interpreter.global_env
        for name, val in std_interpreter.global_env.values.items():
            self.global_env.values[name] = val
        for name, const_flag in std_interpreter.global_env.consts.items():
            self.global_env.consts[name] = const_flag
        for name, typ in std_interpreter.global_env.types.items():
            self.global_env.types[name] = typ

        # Ensure our modules cache includes interpreter's modules
        modules_cache: Dict[str, Environment] = std_interpreter.modules
        # Process imports in order
        for imp in self.program.imports:
            module_name = imp.get('module')
            names = imp.get('names', [])
            # Use interpreter's import_module to load user and std modules
            mod_env = std_interpreter.import_module(module_name, self.global_env)
            for sym in names:
                if sym not in mod_env.values:
                    raise BytieError(ErrorVal('ImportError', f'module {module_name} has no symbol {sym}'))
                # Copy value and its const/type metadata
                self.global_env.values[sym] = mod_env.values[sym]
                if hasattr(mod_env, 'consts') and sym in mod_env.consts:
                    self.global_env.consts[sym] = mod_env.consts[sym]
                if hasattr(mod_env, 'types') and sym in mod_env.types:
                    self.global_env.types[sym] = mod_env.types[sym]

        # Bind placeholders for our program's globals; initially uninitialised
        for idx, name in enumerate(self.program.globals):
            # Avoid clobbering imported names; but allow if imported, spec says re-export? For now, leave imported values
            if name not in self.global_env.values:
                # placeholder NoneVal until assigned
                self.global_env.values[name] = NoneVal()
            # Mark as mutable by default; if compile metadata declares const,
            # we will overwrite this later.
            if name not in self.global_env.consts:
                self.global_env.consts[name] = False
            # Leave type unspecified; compile metadata may fill

        # Install compiled functions into global_env as FunctionValue objects
        for idx, func in enumerate(self.program.functions):
            if func.name:
                # Do not overwrite if imported; store as const
                self.global_env.values[func.name] = FunctionValue(idx, func.params)
                self.global_env.consts[func.name] = True
                # Tag type as Function
                self.global_env.types[func.name] = TypeSpec('Function')

        # Apply compile time global const/type metadata
        if self.program.global_const:
            for name, flag in self.program.global_const.items():
                # delay applying constness until after initial assignment
                self._pending_const[name] = flag
        if self.program.global_types:
            for name, spec_dict in self.program.global_types.items():
                # Convert dict representation back to TypeSpec
                def _typespec_from(d):
                    kind = d['kind']
                    args = d.get('args', [])
                    return TypeSpec(kind, tuple(_typespec_from(a) for a in args))
                self.global_env.types[name] = _typespec_from(spec_dict)

    def run(self) -> Any:
        """Execute the bytecode program and return the value from the entry function."""
        self.link()
        # Execute entry function
        result = self._execute_function(self.program.entry, [])
        # Close debug file
        if self.debug_fp:
            self.debug_fp.close()
        return result

    def _execute_function(self, func_index: int, args: List[Any]) -> Any:
        func = self.program.functions[func_index]
        # Prepare call stack frame
        frame = _Frame(func_index, func.nlocals, return_ip=None)
        # Initialise parameters
        # Parameter values are the first slots in locals
        for i in range(func.params):
            if i < len(args):
                frame.locals[i] = args[i]
            else:
                frame.locals[i] = NoneVal()
        # Value stack for this frame
        stack: List[Any] = []
        code = func.code
        while frame.ip < len(code):
            instr = code[frame.ip]
            op = instr[0]
            # Logging
            if self.debug_level >= 4:
                self._log(f"{frame.ip}: {instr} stack={stack}")
            # Dispatch
            if op == Opcode.NOP:
                pass
            elif op == Opcode.POP:
                if stack:
                    stack.pop()
            elif op == Opcode.DUP:
                if not stack:
                    raise BytieError(ErrorVal('RuntimeError', 'stack underflow on DUP'))
                stack.append(stack[-1])
            elif op == Opcode.ICONST:
                idx = instr[1]
                value = self.program.const_pool.get('ints', [])[idx]
                stack.append(value)
            elif op == Opcode.DCONST:
                idx = instr[1]
                value = self.program.const_pool.get('doubles', [])[idx]
                stack.append(value)
            elif op == Opcode.SCONST:
                idx = instr[1]
                value = self.program.const_pool.get('strings', [])[idx]
                stack.append(value)
            elif op == Opcode.TCONST:
                idx = instr[1]
                # Retrieve TypeSpec constant
                value = None
                if self.program.const_pool is not None:
                    value = self.program.const_pool.get('types', [])[idx]
                stack.append(value)
            elif op == Opcode.TRUE:
                stack.append(1)
            elif op == Opcode.FALSE:
                stack.append(0)
            elif op == Opcode.NULL:
                stack.append(NoneVal())
            elif op == Opcode.LOAD_LOCAL:
                idx = instr[1]
                stack.append(frame.locals[idx])
            elif op == Opcode.STORE_LOCAL:
                idx = instr[1]
                if not stack:
                    raise BytieError(ErrorVal('RuntimeError', 'stack underflow on STORE_LOCAL'))
                value = stack.pop()
                # type check if declared
                # Only enforce type if type information present
                # We'll perform runtime type checks upon assignment to typed locals
                frame.locals[idx] = value
            elif op == Opcode.LOAD_GLOBAL:
                gidx = instr[1]
                name = self.program.globals[gidx]
                print(f"LOAD_GLOBAL: {name} = {self._int   .global_env.values[name]} {stack}")
                if name not in self.global_env.values:
                    raise BytieError(ErrorVal('NameError', f'undefined variable {name}'))
                stack.append(self.global_env.values[name])
                print(f"After LOAD_GLOBAL: {stack=}")
            elif op == Opcode.STORE_GLOBAL:
                gidx = instr[1]
                name = self.program.globals[gidx]
                print(f"STORE_GLOBAL: {name} {stack}")
                if not stack:
                    raise BytieError(ErrorVal('RuntimeError', 'stack underflow on STORE_GLOBAL'))
                value = stack.pop()
                print(f"After STORE_GLOBAL: {value=} {stack=}")
                # Check const flag
                # If pending const assignment, this store acts as declaration.  Delay type check
                if name in self._pending_const:
                    # Do not check existing const flag; will be applied after this assignment
                    pass
                else:
                    if name in self.global_env.consts and self.global_env.consts[name]:
                        raise BytieError(ErrorVal('TypeError', f'cannot assign to const {name}'))
                # Type enforcement
                if name in self.global_env.types:
                    try:
                        # NoneVal as placeholder is not type checked
                        if not isinstance(self.global_env.types[name], TypeSpec):
                            pass
                        else:
                            check_value(value, self.global_env.types[name])
                    except TypeError as e:
                        raise BytieError(ErrorVal('TypeError', str(e)))
                self.global_env.values[name] = value
                # After assignment, apply pending constness if present
                if name in self._pending_const:
                    self.global_env.consts[name] = self._pending_const[name]
                    del self._pending_const[name]
            elif op == Opcode.ADD:
                b = stack.pop()
                a = stack.pop()
                stack.append(self._apply_binary_op('+', a, b))
            elif op == Opcode.SUB:
                b = stack.pop()
                a = stack.pop()
                stack.append(self._apply_binary_op('-', a, b))
            elif op == Opcode.MUL:
                b = stack.pop()
                a = stack.pop()
                stack.append(self._apply_binary_op('*', a, b))
            elif op == Opcode.DIV:
                b = stack.pop()
                a = stack.pop()
                stack.append(self._apply_binary_op('/', a, b))
            elif op == Opcode.MOD:
                b = stack.pop()
                a = stack.pop()
                stack.append(self._apply_binary_op('%', a, b))
            elif op == Opcode.NEG:
                a = stack.pop()
                if isinstance(a, bool):
                    a = int(a)
                if isinstance(a, int) or isinstance(a, float):
                    stack.append(-a)
                else:
                    raise BytieError(ErrorVal('TypeError', f'unary - expects numeric, got {type_name(a)}'))
            elif op == Opcode.NOT:
                a = stack.pop()
                stack.append(0 if self._is_truthy(a) else 1)
            elif op == Opcode.EQ:
                b = stack.pop()
                a = stack.pop()
                eq = self._equal_values(a, b)
                stack.append(1 if eq else 0)
            elif op == Opcode.NE:
                b = stack.pop()
                a = stack.pop()
                eq = self._equal_values(a, b)
                stack.append(0 if eq else 1)
            elif op == Opcode.LT:
                b = stack.pop()
                a = stack.pop()
                stack.append(self._compare(a, b, '<'))
            elif op == Opcode.LE:
                b = stack.pop()
                a = stack.pop()
                stack.append(self._compare(a, b, '<='))
            elif op == Opcode.GT:
                b = stack.pop()
                a = stack.pop()
                stack.append(self._compare(a, b, '>'))
            elif op == Opcode.GE:
                b = stack.pop()
                a = stack.pop()
                stack.append(self._compare(a, b, '>='))
            elif op == Opcode.JUMP:
                target = instr[1]
                frame.ip = target
                continue
            elif op == Opcode.JUMP_IF_FALSE:
                target = instr[1]
                cond = stack.pop()
                if not self._is_truthy(cond):
                    frame.ip = target
                    continue
            elif op == Opcode.RET:
                # return top of stack or None
                if stack:
                    return stack.pop()
                else:
                    return NoneVal()
            elif op == Opcode.LOAD_FUNC:
                fidx = instr[1]
                stack.append(FunctionValue(fidx, self.program.functions[fidx].params))
            elif op == Opcode.CALL:
                argc = instr[1]
                # Pop callee (top of stack) then args
                callee = stack.pop()
                args_vals = []
                for _ in range(argc):
                    if not stack:
                        raise BytieError(ErrorVal('RuntimeError', 'stack underflow on CALL'))
                    args_vals.append(stack.pop())
                # args_vals currently holds arguments from last to first; reverse
                args_vals.reverse()
                if isinstance(callee, BuiltinFunction):
                    try:
                        result = callee.fn(args_vals)
                    except BytieError as ex:
                        err_val = ex.err
                        handled = False
                        while frame.try_stack:
                            handler_ip, depth = frame.try_stack.pop()
                            while len(stack) > depth:
                                stack.pop()
                            stack.append(err_val)
                            frame.ip = handler_ip
                            handled = True
                            break
                        if handled:
                            continue
                        raise ex
                    stack.append(result if result is not None else NoneVal())
                elif isinstance(callee, FunctionValue):
                    if callee.param_count != len(args_vals):
                        ex = BytieError(ErrorVal('TypeError', f'function expects {callee.param_count} arguments'))
                        err_val = ex.err
                        handled = False
                        while frame.try_stack:
                            handler_ip, depth = frame.try_stack.pop()
                            while len(stack) > depth:
                                stack.pop()
                            stack.append(err_val)
                            frame.ip = handler_ip
                            handled = True
                            break
                        if handled:
                            continue
                        raise ex
                    try:
                        result = self._execute_function(callee.func_index, args_vals)
                    except BytieError as ex:
                        err_val = ex.err
                        handled = False
                        while frame.try_stack:
                            handler_ip, depth = frame.try_stack.pop()
                            while len(stack) > depth:
                                stack.pop()
                            stack.append(err_val)
                            frame.ip = handler_ip
                            handled = True
                            break
                        if handled:
                            continue
                        raise ex
                    stack.append(result if result is not None else NoneVal())
                #else:
                #    try:
                #        result = self._interpreter.call_function(callee, args_vals)
                #    except BytieError as ex:
                #        err_val = ex.err
                #        handled = False
                #        while frame.try_stack:
                #            handler_ip, depth = frame.try_stack.pop()
                #            while len(stack) > depth:
                #                stack.pop()
                #            stack.append(err_val)
                #            frame.ip = handler_ip
                #            handled = True
                #            break
                #        if handled:
                #            continue
                #        raise ex
                #    stack.append(result if result is not None else NoneVal())
            elif op == Opcode.BUILD_ARRAY:
                n = instr[1]
                items = []
                for _ in range(n):
                    items.append(stack.pop())
                items.reverse()
                # Infer element type similar to interpreter
                if items:
                    first = items[0]
                    if isinstance(first, ArrayVal):
                        elem_type = TypeSpec('Array', (first.elem_type,))
                    elif isinstance(first, MapVal):
                        elem_type = TypeSpec('Map', (first.value_type,))
                    elif isinstance(first, ErrorVal):
                        elem_type = TypeSpec.error()
                    elif isinstance(first, NoneVal):
                        elem_type = TypeSpec.none()
                    elif isinstance(first, float):
                        elem_type = TypeSpec.double()
                    elif isinstance(first, int):
                        elem_type = TypeSpec.integer()
                    elif isinstance(first, str):
                        elem_type = TypeSpec.string()
                    else:
                        elem_type = TypeSpec(type_name(first))
                else:
                    elem_type = TypeSpec.string()
                stack.append(ArrayVal(elem_type, items))
            elif op == Opcode.BUILD_MAP:
                n = instr[1]
                # pop 2n items: value,key pairs last to first
                entries: Dict[str, Any] = {}
                value_type: Optional[TypeSpec] = None
                for _ in range(n):
                    v = stack.pop()
                    k = stack.pop()
                    # k should be str
                    if not isinstance(k, str):
                        raise BytieError(ErrorVal('TypeError', 'map key must be Str'))
                    entries[k] = v
                    if value_type is None:
                        if isinstance(v, ArrayVal):
                            value_type = TypeSpec('Array', (v.elem_type,))
                        elif isinstance(v, MapVal):
                            value_type = TypeSpec('Map', (v.value_type,))
                        elif isinstance(v, ErrorVal):
                            value_type = TypeSpec.error()
                        elif isinstance(v, NoneVal):
                            value_type = TypeSpec.none()
                        elif isinstance(v, float):
                            value_type = TypeSpec.double()
                        elif isinstance(v, int):
                            value_type = TypeSpec.integer()
                        elif isinstance(v, str):
                            value_type = TypeSpec.string()
                        else:
                            value_type = TypeSpec(type_name(v))
                if value_type is None:
                    value_type = TypeSpec.string()
                stack.append(MapVal(value_type, entries))
            elif op == Opcode.INDEX_GET:
                idx = stack.pop()
                target = stack.pop()
                # Array indexing
                if isinstance(target, ArrayVal):
                    if not isinstance(idx, int):
                        raise BytieError(ErrorVal('TypeError', 'array index must be Integer'))
                    # negative index
                    if idx < 0:
                        idx = len(target.items) + idx
                    if idx < 0 or idx >= len(target.items):
                        raise BytieError(ErrorVal('IndexError', f'array index {idx} out of range'))
                    stack.append(target.items[idx])
                elif isinstance(target, MapVal):
                    if not isinstance(idx, str):
                        raise BytieError(ErrorVal('TypeError', 'map key must be Str'))
                    if idx not in target.entries:
                        raise BytieError(ErrorVal('KeyError', f'key {idx} not found'))
                    stack.append(target.entries[idx])
                else:
                    raise BytieError(ErrorVal('TypeError', f'cannot index {type_name(target)}'))
            elif op == Opcode.INDEX_SET:
                val = stack.pop()
                idx_val = stack.pop()
                target = stack.pop()
                if isinstance(target, ArrayVal):
                    if not isinstance(idx_val, int):
                        raise BytieError(ErrorVal('TypeError', 'array index must be Integer'))
                    if idx_val < 0:
                        idx_val = len(target.items) + idx_val
                    if idx_val < 0 or idx_val >= len(target.items):
                        raise BytieError(ErrorVal('IndexError', f'array index {idx_val} out of range'))
                    # Type check element type
                    try:
                        check_value(val, target.elem_type)
                    except TypeError as e:
                        raise BytieError(ErrorVal('TypeError', str(e)))
                    target.items[idx_val] = val
                    stack.append(val)
                elif isinstance(target, MapVal):
                    if not isinstance(idx_val, str):
                        raise BytieError(ErrorVal('TypeError', 'map key must be Str'))
                    # Type check value type
                    try:
                        check_value(val, target.value_type)
                    except TypeError as e:
                        raise BytieError(ErrorVal('TypeError', str(e)))
                    target.entries[idx_val] = val
                    stack.append(val)
                else:
                    raise BytieError(ErrorVal('TypeError', f'cannot index assign to {type_name(target)}'))
            elif op == Opcode.GET_FIELD:
                field = instr[1]
                target = stack.pop()
                if isinstance(target, MapVal):
                    if field not in target.entries:
                        raise BytieError(ErrorVal('KeyError', f'key {field} not found'))
                    stack.append(target.entries[field])
                elif isinstance(target, ErrorVal):
                    if field == 'name':
                        stack.append(target.name)
                    elif field == 'message':
                        stack.append(target.message)
                    else:
                        raise BytieError(ErrorVal('AttributeError', f'unknown field {field} on Error'))
                else:
                    raise BytieError(ErrorVal('TypeError', f'cannot get field {field} from {type_name(target)}'))
            elif op == Opcode.SET_FIELD:
                field = instr[1]
                val = stack.pop()
                target = stack.pop()
                if isinstance(target, MapVal):
                    # Type check value
                    try:
                        check_value(val, target.value_type)
                    except TypeError as e:
                        raise BytieError(ErrorVal('TypeError', str(e)))
                    target.entries[field] = val
                    stack.append(val)
                elif isinstance(target, ErrorVal):
                    if field == 'name':
                        target.name = val
                        stack.append(val)
                    elif field == 'message':
                        target.message = val
                        stack.append(val)
                    else:
                        raise BytieError(ErrorVal('AttributeError', f'unknown field {field} on Error'))
                else:
                    raise BytieError(ErrorVal('TypeError', f'cannot set field {field} on {type_name(target)}'))
            elif op == Opcode.SETUP_TRY:
                handler_ip = instr[1]
                # record handler address and current stack depth
                self._log(f"setup try handler at {handler_ip}")
                frame.try_stack.append((handler_ip, len(stack)))
            elif op == Opcode.POP_TRY:
                if frame.try_stack:
                    frame.try_stack.pop()
            elif op == Opcode.RAISE:
                # pop error value and raise
                err = stack.pop()
                if not isinstance(err, ErrorVal):
                    raise BytieError(ErrorVal('TypeError', 'RAISE expects Error value'))
                # Unwind to nearest handler
                while frame.try_stack:
                    handler_ip, depth = frame.try_stack.pop()
                    # truncate stack to recorded depth
                    while len(stack) > depth:
                        stack.pop()
                    # push error for handler
                    stack.append(err)
                    frame.ip = handler_ip
                    break
                else:
                    # No handler; propagate BytieError
                    raise BytieError(err)
                continue
            elif op == Opcode.TO_STR:
                # Pop value, push its string representation via to_string helper
                val = stack.pop()
                stack.append(to_string(val))
            else:
                raise BytieError(ErrorVal('RuntimeError', f'unknown opcode {op}'))
            frame.ip += 1
        # If function ends without RET, return None
        return NoneVal()

    # Helper methods replicating interpreter semantics
    def _is_truthy(self, value: Any) -> bool:
        # Copy of Interpreter.is_truthy
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, int):
            return value != 0
        if isinstance(value, float):
            return value != 0.0
        if isinstance(value, str):
            return len(value) > 0
        if isinstance(value, ArrayVal):
            return len(value.items) > 0
        if isinstance(value, MapVal):
            return len(value.entries) > 0
        if isinstance(value, NoneVal):
            return False
        if isinstance(value, ErrorVal):
            return True
        return bool(value)

    def _apply_binary_op(self, op: str, a: Any, b: Any) -> Any:
        # Mirror Interpreter.apply_binary_op
        # String concatenation
        if op == '+':
            if isinstance(a, str) or isinstance(b, str):
                return to_string(a) + to_string(b)
            # numeric addition
            if isinstance(a, float) and isinstance(b, float):
                return a + b
            if isinstance(a, int) and isinstance(b, int):
                return a + b
            if isinstance(a, float) and isinstance(b, int):
                return round_to_int_away_from_zero(a) + b
            if isinstance(a, int) and isinstance(b, float):
                return a + round_to_int_away_from_zero(b)
            raise BytieError(ErrorVal('TypeError', f'unsupported + for {type_name(a)} and {type_name(b)}'))
        if op == '-':
            if isinstance(a, float) and isinstance(b, float):
                return a - b
            if isinstance(a, int) and isinstance(b, int):
                return a - b
            if isinstance(a, float) and isinstance(b, int):
                return round_to_int_away_from_zero(a) - b
            if isinstance(a, int) and isinstance(b, float):
                return a - round_to_int_away_from_zero(b)
            raise BytieError(ErrorVal('TypeError', f'unsupported - for {type_name(a)} and {type_name(b)}'))
        if op == '*':
            if isinstance(a, float) and isinstance(b, float):
                return a * b
            if isinstance(a, int) and isinstance(b, int):
                return a * b
            if isinstance(a, float) and isinstance(b, int):
                return round_to_int_away_from_zero(a) * b
            if isinstance(a, int) and isinstance(b, float):
                return a * round_to_int_away_from_zero(b)
            raise BytieError(ErrorVal('TypeError', f'unsupported * for {type_name(a)} and {type_name(b)}'))
        if op == '/':
            if isinstance(a, float) and isinstance(b, float):
                if b == 0.0:
                    raise BytieError(ErrorVal('RuntimeError', 'division by zero'))
                return a / b
            if isinstance(a, int) and isinstance(b, int):
                if b == 0:
                    raise BytieError(ErrorVal('RuntimeError', 'division by zero'))
                return int(a / b)
            if isinstance(a, float) and isinstance(b, int):
                if b == 0:
                    raise BytieError(ErrorVal('RuntimeError', 'division by zero'))
                return round_to_int_away_from_zero(a) // b
            if isinstance(a, int) and isinstance(b, float):
                if b == 0.0:
                    raise BytieError(ErrorVal('RuntimeError', 'division by zero'))
                return int(a / round_to_int_away_from_zero(b))
            raise BytieError(ErrorVal('TypeError', f'unsupported / for {type_name(a)} and {type_name(b)}'))
        if op == '%':
            if isinstance(a, int) and isinstance(b, int):
                if b == 0:
                    raise BytieError(ErrorVal('RuntimeError', 'modulo by zero'))
                return a % b
            raise BytieError(ErrorVal('TypeError', 'modulo requires Integer operands'))
        raise BytieError(ErrorVal('TypeError', f'unknown operator {op}'))

    def _compare(self, a: Any, b: Any, op: str) -> int:
        # numeric comparisons only; convert float to int using custom rounding
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise BytieError(ErrorVal('TypeError', f'comparison not supported for {type_name(a)} and {type_name(b)}'))
        # align numeric types
        if isinstance(a, float) and isinstance(b, int):
            a = round_to_int_away_from_zero(a)
        if isinstance(a, int) and isinstance(b, float):
            b = round_to_int_away_from_zero(b)
        if op == '<':
            return 1 if a < b else 0
        if op == '<=':
            return 1 if a <= b else 0
        if op == '>':
            return 1 if a > b else 0
        if op == '>=':
            return 1 if a >= b else 0
        raise BytieError(ErrorVal('RuntimeError', f'unknown comparison operator {op}'))

    def _equal_values(self, a: Any, b: Any) -> bool:
        # Mirror Interpreter.equal_values
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if isinstance(a, float) and isinstance(b, int):
                return round_to_int_away_from_zero(a) == b
            if isinstance(a, int) and isinstance(b, float):
                return a == round_to_int_away_from_zero(b)
            return a == b
        if isinstance(a, str) and isinstance(b, str):
            return a == b
        if isinstance(a, ArrayVal) and isinstance(b, ArrayVal):
            if a.elem_type != b.elem_type or len(a.items) != len(b.items):
                return False
            return all(self._equal_values(x, y) for x, y in zip(a.items, b.items))
        if isinstance(a, MapVal) and isinstance(b, MapVal):
            if a.value_type != b.value_type or len(a.entries) != len(b.entries):
                return False
            for k in a.entries:
                if k not in b.entries:
                    return False
                if not self._equal_values(a.entries[k], b.entries[k]):
                    return False
            return True
        if isinstance(a, ErrorVal) and isinstance(b, ErrorVal):
            return a.name == b.name and a.message == b.message
        if isinstance(a, NoneVal) and isinstance(b, NoneVal):
            return True
        return a == b