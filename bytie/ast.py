"""Abstract Syntax Tree (AST) definitions for the Bytie language.

The AST classes defined in this module represent the syntactic structure
of parsed Bytie programs. They are used by the interpreter to evaluate
Bytie code. Each node corresponds to a construct in the Bytie grammar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any

from .types import TypeSpec


@dataclass
class Node:
    """Base class for all AST nodes."""
    pass


@dataclass
class Program(Node):
    body: List[Node]

    def __str__(self) -> str:
        body_str = ", ".join(str(n.__class__.__name__) for n in self.body)
        return f"Program(body={body_str})"

@dataclass
class ImportStmt(Node):
    names: List[str]
    source: str  # module name or string literal (without quotes)


@dataclass
class VarDecl(Node):
    type_spec: TypeSpec
    name: str
    expr: Optional[Node]  # initial value
    is_const: bool = False


@dataclass
class FuncParam:
    type_spec: TypeSpec
    name: str


@dataclass
class FuncDecl(Node):
    name: str
    params: List[FuncParam]
    return_type: TypeSpec
    body: 'Block'

    def __str__(self) -> str:
        body_str = ", ".join(str(n.__class__.__name__) for n in self.body.statements)
        return f"FuncDecl(name={self.name}, params={self.params}, " \
               f"return_type={self.return_type}, body={body_str})"


@dataclass
class Block(Node):
    statements: List[Node]

    def __str__(self) -> str:
        statements_str = ", ".join(str(n.__class__.__name__) for n in self.statements)
        return f"Block(statements={statements_str})"


@dataclass
class IfStmt(Node):
    condition: Node
    then_block: Block
    else_block: Optional[Block]

    def __str__(self) -> str:
        return f"IfStmt(condition={self.condition}, then_block={self.then_block}, else_block={self.else_block})"


@dataclass
class WhileStmt(Node):
    condition: Node
    body: Block

    def __str__(self) -> str:
        return f"WhileStmt(condition={self.condition}, body={self.body})"



@dataclass
class ForStmt(Node):
    init: Optional[Node]  # VarDecl or ExprStmt or None
    condition: Optional[Node]
    post: Optional[Node]
    body: Block

    def __str__(self) -> str:
        return f"ForStmt(init={self.init}, condition={self.condition}, post={self.post}, body={self.body})"


@dataclass
class AttemptStmt(Node):
    try_block: Block
    err_name: str
    catch_block: Block

    def __str__(self) -> str:
        return f"AttemptStmt(try_block={self.try_block}, err_name={self.err_name}, catch_block={self.catch_block})"


@dataclass
class ReturnStmt(Node):
    value: Optional[Node]

@dataclass
class ExprStmt(Node):
    expr: Node


@dataclass
class Assign(Node):
    target: Node  # Ident or Index or Member
    value: Node


@dataclass
class BinaryOp(Node):
    op: str
    left: Node
    right: Node


@dataclass
class UnaryOp(Node):
    op: str
    operand: Node


@dataclass
class Literal(Node):
    value: Any
    literal_type: str  # 'Integer', 'Double', 'Str'


@dataclass
class Ident(Node):
    name: str


@dataclass
class ArrayLit(Node):
    elements: List[Node]


@dataclass
class MapLit(Node):
    entries: List[Tuple[str, Node]]  # keys are string literal values


@dataclass
class Call(Node):
    func: Node
    args: List[Node]


@dataclass
class Index(Node):
    target: Node
    index: Node


@dataclass
class Member(Node):
    target: Node
    name: str