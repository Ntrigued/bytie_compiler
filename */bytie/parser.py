"""Parser for the Bytie language.

This module implements a two-stage parsing pipeline:

1. **Preprocessing**: The raw source code is transformed such that
   newline characters that logically terminate statements are replaced
   with semicolons. This allows us to use a grammar that requires
   semicolons to delimit statements. Comments and strings are respected
   during preprocessing so that newlines inside strings or comments do
   not accidentally terminate statements.

2. **Parsing**: The preprocessed source is fed into a Lark parser
   configured with a grammar for the Bytie language. The resulting
   parse tree is transformed into an abstract syntax tree (AST) using
   a custom transformer.

The `parse_program` function is the public entry point and returns a
`Program` AST node representing the entire source file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple, Optional
import ast as py_ast

from lark import Lark, Transformer, v_args

from .ast import (
    Program, ImportStmt, VarDecl, FuncParam, FuncDecl, Block,
    IfStmt, WhileStmt, ForStmt, AttemptStmt, ReturnStmt, ExprStmt,
    Assign, BinaryOp, UnaryOp, Literal, Ident, ArrayLit, MapLit,
    Call, Index, Member
)
from .types import TypeSpec


def preprocess(source: str) -> str:
    """Insert semicolons at statement boundaries defined by newlines.

    The Bytie language allows optional semicolons at the end of
    statements. To simplify the grammar, we translate newlines that are
    outside of parentheses, brackets, braces, strings, and comments into
    semicolons. This function also preserves the semantics of comments
    by stripping their contents but leaving newlines intact. Strings
    retain their newlines and escapes untouched.
    """
    result: List[str] = []
    depth = 0  # nesting depth for (), [], {}
    i = 0
    length = len(source)
    in_single_quote = False
    in_double_quote = False
    escape = False
    in_line_comment = False
    in_block_comment = False
    while i < length:
        c = source[i]
        # Handle block comments
        if in_block_comment:
            if c == '*' and i + 1 < length and source[i + 1] == '/':
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        # Handle line comments
        if in_line_comment:
            if c == '\n':
                in_line_comment = False
                # newline terminates comment and may terminate statement
                # We will handle newline below
            else:
                i += 1
                continue
        # Detect start of line comment (# or //) when not in string
        if not in_single_quote and not in_double_quote:
            if c == '#' and not in_line_comment:
                in_line_comment = True
                i += 1
                continue
            if c == '/' and i + 1 < length and source[i + 1] == '/':
                in_line_comment = True
                i += 2
                continue
            if c == '/' and i + 1 < length and source[i + 1] == '*':
                in_block_comment = True
                i += 2
                continue
        # Handle strings
        if in_single_quote:
            result.append(c)
            if escape:
                escape = False
            else:
                if c == '\\':
                    escape = True
                elif c == '\'':
                    in_single_quote = False
            i += 1
            continue
        if in_double_quote:
            result.append(c)
            if escape:
                escape = False
            else:
                if c == '\\':
                    escape = True
                elif c == '"':
                    in_double_quote = False
            i += 1
            continue
        # Not inside comment or string
        # Detect entering string
        if c == '\'':
            in_single_quote = True
            result.append(c)
            i += 1
            continue
        if c == '"':
            in_double_quote = True
            result.append(c)
            i += 1
            continue
        # Track nesting depth
        if c in '([{':
            depth += 1
        elif c in ')]}':
            if depth > 0:
                depth -= 1
        # Replace newline by semicolon if at top level
        if c == '\n':
            if depth == 0:
                # Insert semicolon if previous non-whitespace char not semicolon or brace
                # Look back to avoid duplicate semicolons
                # Find last non-whitespace char in result
                j = len(result) - 1
                while j >= 0 and result[j].isspace():
                    j -= 1
                prev = result[j] if j >= 0 else ''
                if prev not in (';', '{', '}'):
                    result.append(';')
            # Always append newline to preserve line numbers (optional)
            # We skip newline char itself as whitespace; skip
            i += 1
            continue
        # Otherwise, copy char
        result.append(c)
        i += 1
    return ''.join(result)


BYTIE_GRAMMAR = r"""
    ?start: program
    ?program: statement*

    // Statements
    ?statement: import_stmt
              | const_decl
              | var_decl
              | func_decl
              | control_stmt
              | attempt_stmt
              | return_stmt
              | expr_stmt
              | empty_stmt

    import_stmt: "retrieve" import_list "from" import_src ";"
    import_list: IDENT ("," IDENT)*
    import_src: IDENT | STRING_LIT

    const_decl: "const" type_spec IDENT "=" expression ";"
    var_decl: type_spec IDENT ["=" expression] ";"

    func_decl: "function" IDENT "(" [param_list] ")" "->" type_spec block
    param_list: param ("," param)*
    param: type_spec IDENT

    control_stmt: if_stmt | while_stmt | for_stmt
    if_stmt: "if" "(" expression ")" block ["else" block]
    while_stmt: "while" "(" expression ")" block
    for_stmt: "for" "(" (var_decl | expr_stmt | ";") expression? ";" expression? ")" block

    attempt_stmt: "attempt" block "fix" "(" IDENT ")" block

    return_stmt: "return" [expression] ";"

    expr_stmt: expression ";"
    empty_stmt: ";"

    block: "{" statement* "}"

    // Expressions with precedence
    ?expression: assign
    ?assign: logic_or ("=" assign)?
    ?logic_or: logic_and ("||" logic_and)*
    ?logic_and: equality ("&&" equality)*
    ?equality: compare (("=="|"!=") compare)*
    ?compare: term (("<"|">"|"<="|">=") term)*
    ?term: factor (("+"|"-") factor)*
    ?factor: unary (("*"|"/"|"%") unary)*
    ?unary: ("!"|"-") unary
          | postfix
    ?postfix: primary
           | postfix "[" expression "]"
           | postfix "." IDENT
           | postfix "(" [arg_list] ")"
    ?primary: literal
            | IDENT
            | "(" expression ")"
            | array_lit
            | map_lit
    literal: INT_LIT | DOUBLE_LIT | STRING_LIT | CHAR_LIT
    array_lit: "[" [expression ("," expression)*] "]"
    map_lit: "{" [map_entry ("," map_entry)*] "}"
    map_entry: STRING_LIT ":" expression
    arg_list: expression ("," expression)*

    // Type specifications
    type_spec: IDENT ["<" type_spec ("," type_spec)* ">"]

    // Tokens
    %import common.WS_INLINE
    %import common.CNAME -> IDENT
    %declare INT_LIT DOUBLE_LIT STRING_LIT CHAR_LIT
    %ignore WS_INLINE

    // Comments
    LINE_COMMENT: /#[^\n]*/ | /\/\/[^\n]*/
    %ignore LINE_COMMENT
    BLOCK_COMMENT: /\/\*[\s\S]*?\*\//
    %ignore BLOCK_COMMENT
"""


BYTIE_PARSER = Lark(
    BYTIE_GRAMMAR,
    parser='lalr',
    propagate_positions=True,
    maybe_placeholders=False,
    lexer='standard',
)


def parse_type_spec(tree) -> TypeSpec:
    """Recursively convert a type_spec parse subtree into a TypeSpec."""
    # tree is of Lark Tree or Token types
    # The grammar defines type_spec: IDENT ["<" type_spec ("," type_spec)* ">"]
    # The resulting tree for type_spec will be: (type_spec IDENT (type_spec...)) or just IDENT token
    from lark import Tree, Token
    if isinstance(tree, Token):
        return TypeSpec(tree.value)
    # tree.data == 'type_spec'
    # first child is IDENT or Tree
    first = tree.children[0]
    if isinstance(first, Token):
        kind = first.value
    else:
        # Unexpected
        kind = str(first)
    # remaining children correspond to type arguments (if any)
    args: List[TypeSpec] = []
    if len(tree.children) > 1:
        for child in tree.children[1:]:
            args.append(parse_type_spec(child))
    return TypeSpec(kind, tuple(args))


class ASTTransformer(Transformer):
    """Transforms the raw parse tree into an AST."""

    def program(self, items):
        return Program(body=list(items))

    def import_stmt(self, items):
        names = []
        # import_list produces list
        import_list_node = items[0]
        if isinstance(import_list_node, list):
            names = import_list_node
        else:
            names = [import_list_node]
        source = items[1]
        # source may include quotes; strip quotes if present
        if source.startswith('"') and source.endswith('"'):
            source = py_ast.literal_eval(source)
        return ImportStmt(names=names, source=source)

    def import_list(self, items):
        return [str(item) for item in items]

    def import_src(self, items):
        token = items[0]
        return str(token)

    def const_decl(self, items):
        type_spec = parse_type_spec(items[0])
        name = str(items[1])
        expr = items[2]
        return VarDecl(type_spec=type_spec, name=name, expr=expr, is_const=True)

    def var_decl(self, items):
        type_spec = parse_type_spec(items[0])
        name = str(items[1])
        expr = items[2] if len(items) > 2 else None
        return VarDecl(type_spec=type_spec, name=name, expr=expr, is_const=False)

    def param_list(self, items):
        return items

    def param(self, items):
        type_spec = parse_type_spec(items[0])
        name = str(items[1])
        return FuncParam(type_spec, name)

    def func_decl(self, items):
        name = str(items[0])
        params: List[FuncParam] = items[1] if isinstance(items[1], list) else []
        return_type = parse_type_spec(items[2])
        body = items[3]
        return FuncDecl(name=name, params=params, return_type=return_type, body=body)

    def block(self, items):
        return Block(statements=list(items))

    def if_stmt(self, items):
        condition = items[0]
        then_block = items[1]
        else_block = items[2] if len(items) > 2 else None
        return IfStmt(condition, then_block, else_block)

    def while_stmt(self, items):
        condition = items[0]
        body = items[1]
        return WhileStmt(condition, body)

    def for_stmt(self, items):
        # items: init, condition, post, body
        init = items[0]
        condition = items[1] if isinstance(items[1], Node) else None
        post = items[2] if isinstance(items[2], Node) else None
        body = items[3]
        return ForStmt(init, condition, post, body)

    def attempt_stmt(self, items):
        try_block = items[0]
        err_name = str(items[1])
        catch_block = items[2]
        return AttemptStmt(try_block, err_name, catch_block)

    def return_stmt(self, items):
        if items:
            value = items[0]
        else:
            value = None
        return ReturnStmt(value)

    def expr_stmt(self, items):
        return ExprStmt(items[0])

    def empty_stmt(self, items):
        return ExprStmt(Literal(None, 'None'))  # no-op

    # Expressions
    def assign(self, items):
        if len(items) == 1:
            return items[0]
        # items[0] = items[1]
        return Assign(target=items[0], value=items[1])

    def binary_expr(self, op, items):
        left = items[0]
        for i in range(1, len(items), 2):
            operator = items[i]
            right = items[i+1]
            left = BinaryOp(op=str(operator), left=left, right=right)
        return left

    def logic_or(self, items):
        # items pattern: expr ( op expr )*
        # But Lark sends tokens and exprs in list; we reconstruct binary left-assoc
        if len(items) == 1:
            return items[0]
        left = items[0]
        i = 1
        while i < len(items):
            op = items[i]
            right = items[i+1]
            left = BinaryOp(op=str(op), left=left, right=right)
            i += 2
        return left

    def logic_and(self, items):
        if len(items) == 1:
            return items[0]
        left = items[0]
        i = 1
        while i < len(items):
            op = items[i]
            right = items[i+1]
            left = BinaryOp(op=str(op), left=left, right=right)
            i += 2
        return left

    def equality(self, items):
        if len(items) == 1:
            return items[0]
        left = items[0]
        i = 1
        while i < len(items):
            op = items[i]
            right = items[i+1]
            left = BinaryOp(op=str(op), left=left, right=right)
            i += 2
        return left

    def compare(self, items):
        if len(items) == 1:
            return items[0]
        left = items[0]
        i = 1
        while i < len(items):
            op = items[i]
            right = items[i+1]
            left = BinaryOp(op=str(op), left=left, right=right)
            i += 2
        return left

    def term(self, items):
        if len(items) == 1:
            return items[0]
        left = items[0]
        i = 1
        while i < len(items):
            op = items[i]
            right = items[i+1]
            left = BinaryOp(op=str(op), left=left, right=right)
            i += 2
        return left

    def factor(self, items):
        if len(items) == 1:
            return items[0]
        left = items[0]
        i = 1
        while i < len(items):
            op = items[i]
            right = items[i+1]
            left = BinaryOp(op=str(op), left=left, right=right)
            i += 2
        return left

    def unary(self, items):
        # unary: ("!"|"-") unary | postfix
        if len(items) == 1:
            return items[0]
        # multiple unary ops could chain, e.g. -!a; apply in order
        op = str(items[0])
        operand = items[1]
        return UnaryOp(op=op, operand=operand)

    def postfix(self, items):
        # Items pattern: base, then chain of postfix operations
        base = items[0]
        i = 1
        while i < len(items):
            op = items[i]
            if op.data == 'index':
                # index: [expression]
                index_expr = items[i].children[0]
                base = Index(target=base, index=index_expr)
                i += 1
            elif op.data == 'member':
                name_token = items[i].children[0]
                base = Member(target=base, name=str(name_token))
                i += 1
            elif op.data == 'call':
                arg_list = items[i].children[0] if items[i].children else []
                args = arg_list if isinstance(arg_list, list) else [arg_list]
                base = Call(func=base, args=args)
                i += 1
            else:
                raise NotImplementedError(f"unsupported postfix op {op.data}")
        return base

    # Lark generates separate rules for postfix patterns; unify them
    def __default__(self, data, children, meta):
        # We need to handle special cases for postfix: index, member, call
        if data == 'index':
            # the parent handles these
            return children[0]
        if data == 'member':
            return children[0]
        if data == 'call':
            return children
        return Transformer.__default__(self, data, children, meta)

    def literal(self, items):
        token = items[0]
        if token.type == 'INT_LIT':
            value = int(token.value)
            return Literal(value, 'Integer')
        if token.type == 'DOUBLE_LIT':
            value = float(token.value)
            return Literal(value, 'Double')
        if token.type == 'STRING_LIT':
            # remove quotes and unescape
            raw = token.value
            # Use Python ast.literal_eval to unescape
            value = py_ast.literal_eval(raw)
            return Literal(value, 'Str')
        if token.type == 'CHAR_LIT':
            raw = token.value  # e.g., '\'a\''
            # Remove single quotes and unescape
            # Create Python string literal representation and eval
            value = py_ast.literal_eval(raw)
            return Literal(value, 'Str')
        raise NotImplementedError(f"unknown literal token {token}")

    def IDENT(self, token):
        return Ident(str(token))

    def array_lit(self, items):
        # items is list of expressions
        return ArrayLit(list(items))

    def map_lit(self, items):
        # items is list of map_entry
        return MapLit(items)

    def map_entry(self, items):
        key_token = items[0]
        key_raw = key_token.value
        key = py_ast.literal_eval(key_raw)
        value = items[1]
        return (key, value)

    def arg_list(self, items):
        return list(items)


def parse_program(source: str) -> Program:
    """Parse Bytie source code into an AST Program.

    The source is first preprocessed to normalize statement terminators.
    Any syntax errors will be raised as exceptions from the parser.
    """
    pre = preprocess(source)
    tree = BYTIE_PARSER.parse(pre)
    ast = ASTTransformer().transform(tree)
    return ast