"""JSON serialization/deserialization for Bytie AST.

This module converts between Bytie AST dataclasses and plain Python
dict/list structures suitable for JSON encoding. It supports a full
round-trip for all node types and `TypeSpec`.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from .ast import (
    Program,
    ImportStmt,
    VarDecl,
    FuncParam,
    FuncDecl,
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
from .types import TypeSpec


def typespec_to_obj(t: TypeSpec) -> Dict[str, Any]:
    return {"kind": t.kind, "args": [typespec_to_obj(a) for a in t.args]}


def typespec_from_obj(o: Dict[str, Any]) -> TypeSpec:
    return TypeSpec(o["kind"], tuple(typespec_from_obj(x) for x in o.get("args", [])))


def ast_to_obj(node: Any) -> Any:
    # Primitives
    if node is None:
        return None
    if isinstance(node, (int, float, str, bool)):
        return node

    # TypeSpec
    if isinstance(node, TypeSpec):
        return {"__type__": "TypeSpec", "value": typespec_to_obj(node)}

    # Node types
    if isinstance(node, Program):
        return {"type": "Program", "body": [ast_to_obj(n) for n in node.body]}
    if isinstance(node, ImportStmt):
        return {"type": "ImportStmt", "names": node.names, "source": node.source}
    if isinstance(node, VarDecl):
        return {
            "type": "VarDecl",
            "type_spec": ast_to_obj(node.type_spec),
            "name": node.name,
            "expr": ast_to_obj(node.expr),
            "is_const": node.is_const,
        }
    if isinstance(node, FuncParam):
        return {
            "type": "FuncParam",
            "type_spec": ast_to_obj(node.type_spec),
            "name": node.name,
        }
    if isinstance(node, FuncDecl):
        return {
            "type": "FuncDecl",
            "name": node.name,
            "params": [ast_to_obj(p) for p in node.params],
            "return_type": ast_to_obj(node.return_type),
            "body": ast_to_obj(node.body),
        }
    if isinstance(node, Block):
        return {"type": "Block", "statements": [ast_to_obj(s) for s in node.statements]}
    if isinstance(node, IfStmt):
        return {
            "type": "IfStmt",
            "condition": ast_to_obj(node.condition),
            "then_block": ast_to_obj(node.then_block),
            "else_block": ast_to_obj(node.else_block),
        }
    if isinstance(node, WhileStmt):
        return {"type": "WhileStmt", "condition": ast_to_obj(node.condition), "body": ast_to_obj(node.body)}
    if isinstance(node, ForStmt):
        return {
            "type": "ForStmt",
            "init": ast_to_obj(node.init),
            "condition": ast_to_obj(node.condition),
            "post": ast_to_obj(node.post),
            "body": ast_to_obj(node.body),
        }
    if isinstance(node, AttemptStmt):
        return {
            "type": "AttemptStmt",
            "try_block": ast_to_obj(node.try_block),
            "err_name": node.err_name,
            "catch_block": ast_to_obj(node.catch_block),
        }
    if isinstance(node, ReturnStmt):
        return {"type": "ReturnStmt", "value": ast_to_obj(node.value)}
    if isinstance(node, ExprStmt):
        return {"type": "ExprStmt", "expr": ast_to_obj(node.expr)}
    if isinstance(node, Assign):
        return {"type": "Assign", "target": ast_to_obj(node.target), "value": ast_to_obj(node.value)}
    if isinstance(node, BinaryOp):
        return {"type": "BinaryOp", "op": node.op, "left": ast_to_obj(node.left), "right": ast_to_obj(node.right)}
    if isinstance(node, UnaryOp):
        return {"type": "UnaryOp", "op": node.op, "operand": ast_to_obj(node.operand)}
    if isinstance(node, Literal):
        return {"type": "Literal", "value": ast_to_obj(node.value), "literal_type": node.literal_type}
    if isinstance(node, Ident):
        return {"type": "Ident", "name": node.name}
    if isinstance(node, ArrayLit):
        return {"type": "ArrayLit", "elements": [ast_to_obj(e) for e in node.elements]}
    if isinstance(node, MapLit):
        return {"type": "MapLit", "entries": [[k, ast_to_obj(v)] for (k, v) in node.entries]}
    if isinstance(node, Call):
        return {"type": "Call", "func": ast_to_obj(node.func), "args": [ast_to_obj(a) for a in node.args]}
    if isinstance(node, Index):
        return {"type": "Index", "target": ast_to_obj(node.target), "index": ast_to_obj(node.index)}
    if isinstance(node, Member):
        return {"type": "Member", "target": ast_to_obj(node.target), "name": node.name}

    raise TypeError(f"Unsupported node for serialization: {type(node).__name__}")


def ast_from_obj(obj: Any) -> Any:
    if obj is None or isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, dict) and obj.get("__type__") == "TypeSpec":
        return typespec_from_obj(obj["value"])
    if not isinstance(obj, dict):
        raise TypeError("Invalid AST object")
    t = obj.get("type")
    if t == "Program":
        return Program(body=[ast_from_obj(n) for n in obj["body"]])
    if t == "ImportStmt":
        return ImportStmt(names=list(obj["names"]), source=obj["source"])
    if t == "VarDecl":
        return VarDecl(
            type_spec=ast_from_obj(obj["type_spec"]),
            name=obj["name"],
            expr=ast_from_obj(obj.get("expr")),
            is_const=bool(obj.get("is_const", False)),
        )
    if t == "FuncParam":
        return FuncParam(type_spec=ast_from_obj(obj["type_spec"]), name=obj["name"])
    if t == "FuncDecl":
        return FuncDecl(
            name=obj["name"],
            params=[ast_from_obj(p) for p in obj["params"]],
            return_type=ast_from_obj(obj["return_type"]),
            body=ast_from_obj(obj["body"]),
        )
    if t == "Block":
        return Block(statements=[ast_from_obj(s) for s in obj["statements"]])
    if t == "IfStmt":
        return IfStmt(
            condition=ast_from_obj(obj["condition"]),
            then_block=ast_from_obj(obj["then_block"]),
            else_block=ast_from_obj(obj.get("else_block")),
        )
    if t == "WhileStmt":
        return WhileStmt(condition=ast_from_obj(obj["condition"]), body=ast_from_obj(obj["body"]))
    if t == "ForStmt":
        return ForStmt(
            init=ast_from_obj(obj.get("init")),
            condition=ast_from_obj(obj.get("condition")),
            post=ast_from_obj(obj.get("post")),
            body=ast_from_obj(obj["body"]),
        )
    if t == "AttemptStmt":
        return AttemptStmt(
            try_block=ast_from_obj(obj["try_block"]),
            err_name=obj["err_name"],
            catch_block=ast_from_obj(obj["catch_block"]),
        )
    if t == "ReturnStmt":
        return ReturnStmt(value=ast_from_obj(obj.get("value")))
    if t == "ExprStmt":
        return ExprStmt(expr=ast_from_obj(obj["expr"]))
    if t == "Assign":
        return Assign(target=ast_from_obj(obj["target"]), value=ast_from_obj(obj["value"]))
    if t == "BinaryOp":
        return BinaryOp(op=obj["op"], left=ast_from_obj(obj["left"]), right=ast_from_obj(obj["right"]))
    if t == "UnaryOp":
        return UnaryOp(op=obj["op"], operand=ast_from_obj(obj["operand"]))
    if t == "Literal":
        return Literal(value=ast_from_obj(obj["value"]), literal_type=obj["literal_type"])
    if t == "Ident":
        return Ident(name=obj["name"])
    if t == "ArrayLit":
        return ArrayLit(elements=[ast_from_obj(e) for e in obj["elements"]])
    if t == "MapLit":
        return MapLit(entries=[(k, ast_from_obj(v)) for (k, v) in obj["entries"]])
    if t == "Call":
        return Call(func=ast_from_obj(obj["func"]), args=[ast_from_obj(a) for a in obj["args"]])
    if t == "Index":
        return Index(target=ast_from_obj(obj["target"]), index=ast_from_obj(obj["index"]))
    if t == "Member":
        return Member(target=ast_from_obj(obj["target"]), name=obj["name"])

    raise ValueError(f"Unknown AST node type: {t}")


