"""
Bytecode lowering for the Bytie language.

This module takes an abstract syntax tree (AST) of a Bytie program and
produces a BBC1 bytecode program.  The lowering is performed in a
single pass over the AST.  The emitter tracks lexical scopes to
distinguish between local and global variables, builds a constant
pool for literals, records imports, and emits an instruction stream
for each function.  Currently only a minimal subset of the language is
supported; additional nodes and features will be lowered as they are
required by the tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from .ast_json import ast_from_obj
from .ast import (
    Program,
    ImportStmt,
    VarDecl,
    FuncDecl,
    FuncParam,
    Block,
    IfStmt,
    WhileStmt,
    ForStmt,
    AttemptStmt,
    ReturnStmt,
    ExprStmt,
    Assign,
    BinaryOp,
    UnaryOp,
    Literal,
    Ident,
    ArrayLit,
    MapLit,
    Call,
    Index,
    Member,
)
from .types import TypeSpec, NoneVal, ArrayVal, MapVal
from .bytecode import (
    BytecodeProgram,
    BytecodeFunction,
    Opcode,
)


class Scope:
    """Represents a function scope with local variables and parameters."""

    def __init__(self, parent: Optional['Scope'], func: Optional[FuncDecl] = None):
        self.parent = parent
        self.func = func
        # mapping from variable name to local index
        self.locals: Dict[str, int] = {}
        # list of local names in order of index
        self.local_names: List[str] = []
        # number of parameters
        self.param_count: int = 0
        if func is not None:
            self.param_count = len(func.params)
            for i, p in enumerate(func.params):
                self.locals[p.name] = i
                self.local_names.append(p.name)

    def declare_local(self, name: str) -> int:
        if name in self.locals:
            return self.locals[name]
        idx = len(self.local_names)
        self.locals[name] = idx
        self.local_names.append(name)
        return idx

    def resolve(self, name: str) -> Optional[int]:
        """Return local index if name is local to this scope, else None."""
        if name in self.locals:
            return self.locals[name]
        return None


class Emitter:
    """Lowers an AST into bytecode."""

    def __init__(self, ast_root: Program):
        self.ast_root = ast_root
        self.const_pool: Dict[str, List[Any]] = {
            'ints': [],
            'doubles': [],
            'strings': [],
        }
        # map literal value to index in const pool for deduplication
        self.int_map: Dict[int, int] = {}
        self.double_map: Dict[float, int] = {}
        self.string_map: Dict[str, int] = {}
        # list of import records
        self.imports: List[Dict[str, Any]] = []
        # global symbol table: list of names; maintain insertion order
        self.globals: List[str] = []
        # mapping from global name to index
        self.global_map: Dict[str, int] = {}
        # functions list
        self.functions: List[BytecodeFunction] = []
        # const and type metadata for globals
        self.global_consts: Dict[str, bool] = {}
        self.global_types: Dict[str, Dict[str, Any]] = {}
        # type constant pool
        self.type_pool: List[Any] = []
        self.type_map: Dict[str, int] = {}

    def compile(self) -> BytecodeProgram:
        """Perform lowering and return a BytecodeProgram."""
        # Process top‑level statements and gather function definitions
        # Emit main function
        main_code: List[List[Any]] = []
        # Top‑level scope: no locals, but used for resolving global names
        global_scope = Scope(None)
        # Walk program body
        for stmt in self.ast_root.body:
            if isinstance(stmt, ImportStmt):
                # Record import; no code emitted
                self.imports.append({'module': stmt.source, 'names': list(stmt.names)})
                # Imported names are considered globals; ensure they appear in global list
                for name in stmt.names:
                    self._ensure_global(name)
            elif isinstance(stmt, FuncDecl):
                # Compile function separately
                self._compile_function(stmt)
                # Register function name as global
                self._ensure_global(stmt.name)
            else:
                # Other statements lowered into main
                self._emit_statement(stmt, main_code, global_scope, in_function=False)
        # End of main: return None
        main_code.append([Opcode.RET])
        # Create main function (entry)
        main_func = BytecodeFunction(name='main', params=0, locals=[], nlocals=0, code=main_code)
        # Prepend main to functions list
        entry_index = 0
        self.functions.insert(0, main_func)
        # Compose program
        program = BytecodeProgram(
            magic='BYTC',
            version=1,
            const_pool=self.const_pool,
            imports=self.imports,
            globals=self.globals,
            functions=self.functions,
            entry=entry_index,
        )
        # Attach type constants into program const_pool
        if self.type_pool:
            program.const_pool['types'] = self.type_pool
        if self.global_consts:
            program.global_const = self.global_consts
        if self.global_types:
            program.global_types = self.global_types
        return program

    def _ensure_global(self, name: str) -> int:
        """Add a name to the global symbol table if not present; return its index."""
        if name not in self.global_map:
            idx = len(self.globals)
            self.global_map[name] = idx
            self.globals.append(name)
        return self.global_map[name]

    def _add_const_int(self, value: int) -> int:
        if value in self.int_map:
            return self.int_map[value]
        idx = len(self.const_pool['ints'])
        self.const_pool['ints'].append(value)
        self.int_map[value] = idx
        return idx

    def _add_const_double(self, value: float) -> int:
        if value in self.double_map:
            return self.double_map[value]
        idx = len(self.const_pool['doubles'])
        self.const_pool['doubles'].append(value)
        self.double_map[value] = idx
        return idx

    def _add_const_string(self, value: str) -> int:
        if value in self.string_map:
            return self.string_map[value]
        idx = len(self.const_pool['strings'])
        self.const_pool['strings'].append(value)
        self.string_map[value] = idx
        return idx

    def _add_const_type(self, typ: TypeSpec) -> int:
        """Add a TypeSpec constant to the pool and return its index."""
        # Use repr or a stable key
        key = repr(typ)
        if key in self.type_map:
            return self.type_map[key]
        idx = len(self.type_pool)
        self.type_pool.append(typ)
        self.type_map[key] = idx
        return idx

    def _compile_function(self, func_decl: FuncDecl):
        """Compile a function declaration into a BytecodeFunction and add it to self.functions."""
        scope = Scope(None, func_decl)
        code: List[List[Any]] = []
        # Compile function body
        self._emit_statement(func_decl.body, code, scope, in_function=True)
        # Ensure a return at end
        code.append([Opcode.RET])
        bc_func = BytecodeFunction(
            name=func_decl.name,
            params=len(func_decl.params),
            locals=scope.local_names,
            nlocals=len(scope.local_names),
            code=code,
        )
        self.functions.append(bc_func)

    def _emit_statement(self, node: Any, code: List[List[Any]], scope: Scope, in_function: bool):
        # Import statements inside any scope simply record the import and ensure globals
        if isinstance(node, ImportStmt):
            # Record import; no code emitted
            self.imports.append({'module': node.source, 'names': list(node.names)})
            for name in node.names:
                self._ensure_global(name)
            return
        """Emit bytecode instructions for a statement or block."""
        # Blocks create new local scope for statements
        if isinstance(node, Block):
            # New nested scope inherits locals but does not shadow outer locals in this first pass
            for stmt in node.statements:
                self._emit_statement(stmt, code, scope, in_function)
            return
        if isinstance(node, VarDecl):
            # Evaluate initializer if present
            if node.expr is not None:
                self._emit_expression(node.expr, code, scope, in_function)
            else:
                # push default value onto stack (for now, push None)
                code.append([Opcode.NULL])
            # Determine if this is global (top level) or local (inside function)
            if in_function:
                idx = scope.declare_local(node.name)
                code.append([Opcode.STORE_LOCAL, idx])
            else:
                gidx = self._ensure_global(node.name)
                code.append([Opcode.STORE_GLOBAL, gidx])
                # Record const and type metadata
                self.global_consts[node.name] = bool(node.is_const)
                # Serialise TypeSpec to dict
                if node.type_spec is not None:
                    def _typespec_to(d):
                        return {
                            'kind': d.kind,
                            'args': [_typespec_to(a) for a in d.args],
                        }
                    self.global_types[node.name] = _typespec_to(node.type_spec)
            # Ignore type and const for now; runtime checks performed in VM
            return
        if isinstance(node, ExprStmt):
            self._emit_expression(node.expr, code, scope, in_function)
            # Pop unused result
            code.append([Opcode.POP])
            return
        if isinstance(node, Assign):
            # Assignment in statement context: evaluate value and perform store, discarding result
            # Evaluate right-hand side
            self._emit_expression(node.value, code, scope, in_function)
            # Discard result after assignment
            # Determine lvalue target
            target = node.target
            if isinstance(target, Ident):
                # Assign to local or global
                local_idx = scope.resolve(target.name) if in_function else None
                if local_idx is not None:
                    code.append([Opcode.STORE_LOCAL, local_idx])
                else:
                    gidx = self._ensure_global(target.name)
                    code.append([Opcode.STORE_GLOBAL, gidx])
            elif isinstance(target, Index):
                # Evaluate index and target and perform INDEX_SET
                self._emit_expression(target.index, code, scope, in_function)
                self._emit_expression(target.target, code, scope, in_function)
                code.append([Opcode.INDEX_SET])
            elif isinstance(target, Member):
                # The interpreter does not allow assignment to properties; raise error
                raise NotImplementedError("Assignment to member is not supported")
            else:
                raise NotImplementedError(f"Unsupported assignment target: {type(target)}")
            # For index assignment, the INDEX_SET instruction leaves the value on stack; pop it
            if isinstance(target, Index):
                code.append([Opcode.POP])
            # For Ident assignment, STORE_* consumes the value so no extra pop needed
            return
        if isinstance(node, IfStmt):
            # Evaluate condition
            self._emit_expression(node.condition, code, scope, in_function)
            # Placeholder for jump if false; record index
            jmp_false_idx = len(code)
            code.append([Opcode.JUMP_IF_FALSE, None])  # to be filled
            # then block
            self._emit_statement(node.then_block, code, scope, in_function)
            # Jump to end
            jmp_end_idx = len(code)
            code.append([Opcode.JUMP, None])
            # else block target
            else_target = len(code)
            if node.else_block is not None:
                self._emit_statement(node.else_block, code, scope, in_function)
            # fill in jump targets
            code[jmp_false_idx][1] = else_target
            end_target = len(code)
            code[jmp_end_idx][1] = end_target
            return
        if isinstance(node, WhileStmt):
            # loop header position
            loop_start = len(code)
            # condition
            self._emit_expression(node.condition, code, scope, in_function)
            # jump out if false
            jmp_out_idx = len(code)
            code.append([Opcode.JUMP_IF_FALSE, None])
            # body
            self._emit_statement(node.body, code, scope, in_function)
            # jump back to start
            code.append([Opcode.JUMP, loop_start])
            out_target = len(code)
            code[jmp_out_idx][1] = out_target
            return
        if isinstance(node, ForStmt):
            # init (may be VarDecl or ExprStmt)
            if node.init is not None:
                if isinstance(node.init, VarDecl):
                    # new variable local to this loop; treat as local
                    self._emit_statement(node.init, code, scope, in_function)
                elif isinstance(node.init, ExprStmt):
                    self._emit_expression(node.init.expr, code, scope, in_function)
                    code.append([Opcode.POP])
            loop_cond_pos = len(code)
            # condition
            if node.condition is not None:
                self._emit_expression(node.condition, code, scope, in_function)
            else:
                # no condition means always true; push true
                code.append([Opcode.TRUE])
            jmp_out_idx = len(code)
            code.append([Opcode.JUMP_IF_FALSE, None])
            # body
            self._emit_statement(node.body, code, scope, in_function)
            # post
            if node.post is not None:
                # post can be VarDecl, Assign, ExprStmt; treat as statement
                if isinstance(node.post, VarDecl) or isinstance(node.post, Assign) or isinstance(node.post, ExprStmt):
                    # Use statement emitter to handle assignment semantics
                    self._emit_statement(node.post, code, scope, in_function)
                else:
                    # Evaluate expression and discard
                    self._emit_expression(node.post, code, scope, in_function)
                    code.append([Opcode.POP])
            # jump back to condition
            code.append([Opcode.JUMP, loop_cond_pos])
            out_target = len(code)
            code[jmp_out_idx][1] = out_target
            return
        if isinstance(node, ReturnStmt):
            if node.value is not None:
                self._emit_expression(node.value, code, scope, in_function)
            else:
                code.append([Opcode.NULL])
            code.append([Opcode.RET])
            return
        if isinstance(node, AttemptStmt):
            # try/catch
            # SETUP_TRY with handler target
            handler_placeholder = len(code)
            code.append([Opcode.SETUP_TRY, None])
            # try block
            self._emit_statement(node.try_block, code, scope, in_function)
            # POP_TRY after normal completion
            code.append([Opcode.POP_TRY])
            # Jump past catch
            end_jmp_placeholder = len(code)
            code.append([Opcode.JUMP, None])
            # handler label
            handler_target = len(code)
            code[handler_placeholder][1] = handler_target
            # store error into catch var (local to this scope)
            # On handler entry, error is on stack
            if in_function:
                idx = scope.declare_local(node.err_name)
                # store error
                code.append([Opcode.STORE_LOCAL, idx])
            else:
                gidx = self._ensure_global(node.err_name)
                code.append([Opcode.STORE_GLOBAL, gidx])
            # catch block
            self._emit_statement(node.catch_block, code, scope, in_function)
            # POP_TRY after catch
            code.append([Opcode.POP_TRY])
            # fill end jump target
            end_target = len(code)
            code[end_jmp_placeholder][1] = end_target
            return
        # For unsupported node types, raise NotImplementedError
        raise NotImplementedError(f"Unsupported statement: {type(node).__name__}")

    def _emit_expression(self, node: Any, code: List[List[Any]], scope: Scope, in_function: bool):
        """Emit bytecode instructions for an expression, pushing its value on stack."""
        if isinstance(node, Literal):
            val = node.value
            if isinstance(val, bool):
                code.append([Opcode.TRUE if val else Opcode.FALSE])
            elif isinstance(val, int):
                idx = self._add_const_int(val)
                code.append([Opcode.ICONST, idx])
            elif isinstance(val, float):
                idx = self._add_const_double(val)
                code.append([Opcode.DCONST, idx])
            elif isinstance(val, str):
                idx = self._add_const_string(val)
                code.append([Opcode.SCONST, idx])
            elif isinstance(val, NoneVal):
                code.append([Opcode.NULL])
            elif isinstance(val, TypeSpec):
                # Push full TypeSpec constant
                idx = self._add_const_type(val)
                code.append([Opcode.TCONST, idx])
            else:
                # Some literal values may be dictionaries representing TypeSpec
                if isinstance(val, dict) and 'kind' in val:
                    # Convert dict to TypeSpec
                    def _dict_to_typespec(d):
                        kind = d['kind']
                        args = d.get('args', [])
                        return TypeSpec(kind, tuple(_dict_to_typespec(a) if isinstance(a, dict) else a for a in args))
                    ts = _dict_to_typespec(val)
                    idx = self._add_const_type(ts)
                    code.append([Opcode.TCONST, idx])
                else:
                    raise NotImplementedError(f"Unsupported literal value: {val}")
            return
        if isinstance(node, Ident):
            # Handle builtin type names (and synonyms) as TypeSpec constants.
            # In the interpreter, identifiers like Integer, Double, Str, Array, Map, Error, None
            # are treated as type literals when passed to convert_t().  Also Int and String are
            # synonyms for Integer and Str respectively.
            synonyms = {
                'Int': 'Integer',
                'String': 'Str',
            }
            canonical = synonyms.get(node.name, node.name)
            builtin_types = {'Integer', 'Double', 'Str', 'Array', 'Map', 'Error', 'None'}
            if canonical in builtin_types:
                # push a TypeSpec constant
                spec = TypeSpec(canonical)
                idx = self._add_const_type(spec)
                code.append([Opcode.TCONST, idx])
                return
            # Otherwise, resolve identifier as local or global
            local_idx = scope.resolve(node.name) if in_function else None
            if local_idx is not None:
                code.append([Opcode.LOAD_LOCAL, local_idx])
            else:
                gidx = self._ensure_global(node.name)
                code.append([Opcode.LOAD_GLOBAL, gidx])
            return
        if isinstance(node, UnaryOp):
            # Evaluate operand
            self._emit_expression(node.operand, code, scope, in_function)
            if node.op == '!':
                code.append([Opcode.NOT])
            elif node.op == '-':
                code.append([Opcode.NEG])
            else:
                raise NotImplementedError(f"Unsupported unary op {node.op}")
            return
        if isinstance(node, BinaryOp):
            if node.op in ('&&', '||'):
                # Short‑circuit logical operators
                if node.op == '&&':
                    # Evaluate left operand
                    self._emit_expression(node.left, code, scope, in_function)
                    # If left is false, jump to false branch
                    jmp_left_false = len(code)
                    code.append([Opcode.JUMP_IF_FALSE, None])
                    # Left is true: evaluate right
                    self._emit_expression(node.right, code, scope, in_function)
                    # If right is false, jump to false branch
                    jmp_right_false = len(code)
                    code.append([Opcode.JUMP_IF_FALSE, None])
                    # Both true -> push TRUE
                    code.append([Opcode.TRUE])
                    # Jump to end
                    jmp_end = len(code)
                    code.append([Opcode.JUMP, None])
                    # False branch label
                    false_label = len(code)
                    code.append([Opcode.FALSE])
                    # End label
                    end_label = len(code)
                    # Patch
                    code[jmp_left_false][1] = false_label
                    code[jmp_right_false][1] = false_label
                    code[jmp_end][1] = end_label
                else:  # '||'
                    # Evaluate left operand
                    self._emit_expression(node.left, code, scope, in_function)
                    # If left is truthy, jump to true branch
                    jmp_eval_right = len(code)
                    code.append([Opcode.JUMP_IF_FALSE, None])
                    # Left truthy: push TRUE
                    code.append([Opcode.TRUE])
                    jmp_end1 = len(code)
                    code.append([Opcode.JUMP, None])
                    # Label: evaluate right operand
                    eval_right_label = len(code)
                    # Patch to jump to evaluate right
                    code[jmp_eval_right][1] = eval_right_label
                    # Evaluate right
                    self._emit_expression(node.right, code, scope, in_function)
                    # If right false -> false branch
                    jmp_right_false = len(code)
                    code.append([Opcode.JUMP_IF_FALSE, None])
                    # Right truthy: push TRUE
                    code.append([Opcode.TRUE])
                    jmp_end2 = len(code)
                    code.append([Opcode.JUMP, None])
                    # False branch label
                    false_label = len(code)
                    code.append([Opcode.FALSE])
                    # End label
                    end_label = len(code)
                    # Patch second level jumps
                    code[jmp_right_false][1] = false_label
                    code[jmp_end1][1] = end_label
                    code[jmp_end2][1] = end_label
                return
            else:
                # Evaluate left then right
                self._emit_expression(node.left, code, scope, in_function)
                self._emit_expression(node.right, code, scope, in_function)
                op_map = {
                    '+': Opcode.ADD,
                    '-': Opcode.SUB,
                    '*': Opcode.MUL,
                    '/': Opcode.DIV,
                    '%': Opcode.MOD,
                    '==': Opcode.EQ,
                    '!=': Opcode.NE,
                    '<': Opcode.LT,
                    '<=': Opcode.LE,
                    '>': Opcode.GT,
                    '>=': Opcode.GE,
                }
                if node.op not in op_map:
                    raise NotImplementedError(f"Unsupported binary op {node.op}")
                code.append([op_map[node.op]])
                return
        if isinstance(node, Call):
            # Emit args first
            for arg in node.args:
                self._emit_expression(arg, code, scope, in_function)
            # Emit callee
            self._emit_expression(node.func, code, scope, in_function)
            code.append([Opcode.CALL, len(node.args)])
            return
        if isinstance(node, ArrayLit):
            # Emit elements left to right
            for el in node.elements:
                self._emit_expression(el, code, scope, in_function)
            code.append([Opcode.BUILD_ARRAY, len(node.elements)])
            return
        if isinstance(node, MapLit):
            # Each entry: key (string) and value expression
            # To preserve insertion order when popping, push entries in reverse order
            for key, value_node in reversed(node.entries):
                # push key constant
                idx = self._add_const_string(key)
                code.append([Opcode.SCONST, idx])
                # push value
                self._emit_expression(value_node, code, scope, in_function)
            code.append([Opcode.BUILD_MAP, len(node.entries)])
            return
        if isinstance(node, Index):
            # Evaluate target first, then index. INDEX_GET will pop index then target.
            self._emit_expression(node.target, code, scope, in_function)
            self._emit_expression(node.index, code, scope, in_function)
            code.append([Opcode.INDEX_GET])
            return
        if isinstance(node, Member):
            self._emit_expression(node.target, code, scope, in_function)
            code.append([Opcode.GET_FIELD, node.name])
            return
        if isinstance(node, AttemptStmt):
            # attempt expression returns value of try or catch; we treat similarly to statement but push result
            raise NotImplementedError('AttemptStmt in expression context not supported yet')
        # Assignment can appear in expression context
        if isinstance(node, Assign):
            # Evaluate RHS
            self._emit_expression(node.value, code, scope, in_function)
            target = node.target
            # Duplicate value for Ident assignment to keep a copy on stack
            if isinstance(target, Ident):
                # Duplicate top-of-stack value
                code.append([Opcode.DUP])
                # Now store one copy
                local_idx = scope.resolve(target.name) if in_function else None
                if local_idx is not None:
                    code.append([Opcode.STORE_LOCAL, local_idx])
                else:
                    gidx = self._ensure_global(target.name)
                    code.append([Opcode.STORE_GLOBAL, gidx])
                # The duplicate remains on stack as expression result
            elif isinstance(target, Index):
                # For index assignment, we must preserve evaluation order: value is evaluated first,
                # but INDEX_SET expects stack order target,index,value (bottom→top before pop sequence).
                # To achieve this, store value into a temporary local, evaluate target and index, then
                # reload value.
                # Create a new temporary local slot
                tmp_name = f"$tmp{len(scope.local_names)}"
                tmp_idx = scope.declare_local(tmp_name)
                # Store the value into temp (pops from stack)
                code.append([Opcode.STORE_LOCAL, tmp_idx])
                # Evaluate target (container)
                self._emit_expression(target.target, code, scope, in_function)
                # Evaluate index
                self._emit_expression(target.index, code, scope, in_function)
                # Load value back
                code.append([Opcode.LOAD_LOCAL, tmp_idx])
                # Now stack: target, index, value
                code.append([Opcode.INDEX_SET])
            elif isinstance(target, Member):
                # Assignment to member is unsupported in interpreter
                raise NotImplementedError("Assignment to member is not supported")
            else:
                raise NotImplementedError(f"Unsupported assignment target: {type(target)}")
            return
        raise NotImplementedError(f"Unsupported expression: {type(node).__name__}")