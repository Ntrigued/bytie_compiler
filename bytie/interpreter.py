"""Interpreter and compiler for the Bytie language.

This module implements the complete Bytie toolchain: a tokenizer, a
recursive‑descent parser producing an AST, and an interpreter that
evaluates Bytie programs. The language is defined by the grammar and
runtime semantics described in the specification. See README.md for
usage.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import pathlib
import builtins

from .std.io import populate_io_environment
from .types import (
    TypeSpec, NoneVal, ErrorVal, ArrayVal, MapVal,
    round_to_int_away_from_zero, check_value, convert_value, to_string, type_name,
)
from .ast import (
    Program, ImportStmt, VarDecl, FuncParam, FuncDecl, Block,
    IfStmt, WhileStmt, ForStmt, AttemptStmt, ReturnStmt, ExprStmt,
    Assign, BinaryOp, UnaryOp, Literal, Ident, ArrayLit, MapLit,
    Call, Index, Member, Node
)
from .errors import BytieError, ReturnSignal
from .enviironment import Environment
from .builtin_function import BuiltinFunction

###############################################################################
# Tokenizer
###############################################################################

@dataclass
class Token:
    type: str
    value: str
    line: int
    column: int


class LexerError(Exception):
    pass


def preprocess(source: str) -> str:
    """Insert semicolons at logical statement boundaries.

    Newlines outside parentheses, brackets, braces, strings, and comments
    terminate statements. This function replaces those newlines with
    semicolons so that the parser can rely solely on semicolons to
    delimit statements. Comments are removed. Strings retain their
    newlines and escapes.
    """
    result: List[str] = []
    depth = 0
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
            i += 1
            # skip the comment content entirely
            continue
        # Detect start of comments when not in string
        if not in_single_quote and not in_double_quote:
            if c == '#' or (c == '/' and i + 1 < length and source[i + 1] == '/'):
                in_line_comment = True
                # skip both slashes for //
                if c == '/' and source[i + 1] == '/':
                    i += 2
                else:
                    i += 1
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
        # Start of string
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
        # Track parentheses/braces/brackets
        if c in '([{':
            depth += 1
        elif c in ')]}':
            if depth > 0:
                depth -= 1
        # Handle newline
        if c == '\n':
            if depth == 0:
                # Insert semicolon if not already preceded by semicolon or brace
                j = len(result) - 1
                while j >= 0 and result[j].isspace():
                    j -= 1
                prev = result[j] if j >= 0 else ''
                if prev not in (';', '{', '}'):
                    result.append(';')
            # newlines dropped otherwise
            i += 1
            continue
        # copy other characters
        result.append(c)
        i += 1
    return ''.join(result)


def tokenize(source: str) -> List[Token]:
    """Convert source code into a list of tokens.

    The lexer recognizes identifiers, numbers, string and char literals,
    keywords, operators, and punctuation. Comments should already be
    removed by the preprocessor. The minus sign is always tokenized as
    a separate operator; negative numbers are parsed by the unary
    expression handler.
    """
    tokens: List[Token] = []
    i = 0
    line = 1
    col = 1
    length = len(source)

    def advance(n: int = 1):
        nonlocal i, col, line
        for _ in range(n):
            if i < length and source[i] == '\n':
                line += 1
                col = 1
            else:
                col += 1
            i += 1

    while i < length:
        c = source[i]
        # Skip whitespace
        if c.isspace():
            advance()
            continue
        # Identifiers or keywords
        if c.isalpha() or c == '_':
            start_col = col
            start_i = i
            while i < length and (source[i].isalnum() or source[i] == '_'):
                advance()
            value = source[start_i:i]
            tokens.append(Token('IDENT', value, line, start_col))
            continue
        # Numbers (integer or double)
        if c.isdigit():
            start_col = col
            start_i = i
            has_dot = False
            while i < length and (source[i].isdigit() or source[i] == '.'):
                if source[i] == '.':
                    if has_dot:
                        raise LexerError(f"unexpected second '.' in number at {line}:{col}")
                    has_dot = True
                advance()
            value = source[start_i:i]
            if has_dot:
                tokens.append(Token('DOUBLE', value, line, start_col))
            else:
                tokens.append(Token('INT', value, line, start_col))
            continue
        # String literal
        if c == '"':
            start_col = col
            advance()  # skip opening quote
            raw_chars: List[str] = []
            escape = False
            while i < length:
                ch = source[i]
                if escape:
                    # Accept any escaped char; Python's escapes will be processed later
                    raw_chars.append('\\' + ch)
                    escape = False
                    advance()
                    continue
                if ch == '\\':
                    escape = True
                    advance()
                    continue
                if ch == '"':
                    advance()  # skip closing quote
                    break
                # regular char
                raw_chars.append(ch)
                advance()
            else:
                raise LexerError(f"unterminated string literal at {line}:{start_col}")
            # Create Python literal syntax: wrap in quotes
            py_lit = '"' + ''.join(raw_chars) + '"'
            tokens.append(Token('STRING', py_lit, line, start_col))
            continue
        # Char literal
        if c == '\'':
            start_col = col
            advance()
            raw_chars: List[str] = []
            escape = False
            while i < length:
                ch = source[i]
                if escape:
                    raw_chars.append('\\' + ch)
                    escape = False
                    advance()
                    continue
                if ch == '\\':
                    escape = True
                    advance()
                    continue
                if ch == '\'':
                    advance()
                    break
                raw_chars.append(ch)
                advance()
            else:
                raise LexerError(f"unterminated char literal at {line}:{start_col}")
            py_lit = '\'' + ''.join(raw_chars) + '\''
            tokens.append(Token('CHAR', py_lit, line, start_col))
            continue
        # Multi-character operators
        two_char_ops = {'==', '!=', '>=', '<=', '&&', '||', '->'}
        if i + 1 < length:
            pair = source[i:i+2]
            if pair in two_char_ops:
                tokens.append(Token(pair, pair, line, col))
                advance(2)
                continue
        # Single-character operators and punctuation
        single_ops = {'=', '<', '>', '+', '-', '*', '/', '%', '!', '[', ']', '{', '}', '(', ')', ';', ',', '.', ':', '<', '>'}
        if c in single_ops:
            tokens.append(Token(c, c, line, col))
            advance()
            continue
        raise LexerError(f"unexpected character {c!r} at {line}:{col}")
    return tokens


###############################################################################
# Parser implementation
###############################################################################


class ParseError(Exception):
    pass


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[Token]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, expected: Union[str, List[str]]) -> Token:
        token = self.peek()
        if token is None:
            raise ParseError(f"unexpected end of input, expected {expected}")
        if isinstance(expected, list):
            if token.type not in expected and token.value not in expected:
                raise ParseError(f"expected one of {expected} at {token.line}:{token.column}, got {token.type} {token.value}")
        else:
            if token.type != expected and token.value != expected:
                raise ParseError(f"expected {expected} at {token.line}:{token.column}, got {token.type} {token.value}")
        self.pos += 1
        return token

    def match(self, expected: Union[str, List[str]]) -> bool:
        token = self.peek()
        if token is None:
            return False
        if isinstance(expected, list):
            return token.type in expected or token.value in expected
        return token.type == expected or token.value == expected

    def parse_program(self) -> Program:
        statements: List[Node] = []
        while self.peek() is not None:
            # if only semicolons left, consume them
            if self.match(';'):
                self.consume(';')
                continue
            stmt = self.parse_statement()
            statements.append(stmt)
        return Program(statements)

    def parse_statement(self) -> Node:
        token = self.peek()
        if token is None:
            raise ParseError("unexpected end of input")
        # import
        if token.value == 'retrieve':
            return self.parse_import_stmt()
        # const decl
        if token.value == 'const':
            return self.parse_const_decl()
        # function
        if token.value == 'function':
            return self.parse_func_decl()
        # attempt
        if token.value == 'attempt':
            return self.parse_attempt_stmt()
        # if
        if token.value == 'if':
            return self.parse_if_stmt()
        # while
        if token.value == 'while':
            return self.parse_while_stmt()
        # for
        if token.value == 'for':
            return self.parse_for_stmt()
        # return
        if token.value == 'return':
            return self.parse_return_stmt()
        # block
        if token.value == '{':
            return self.parse_block()
        # var decl (type spec) or expr stmt
        # If next token is a type keyword or IDENT that is a type spec, treat as var decl
        if self.is_type_spec_start():
            return self.parse_var_decl()
        # else expression statement
        expr = self.parse_expression()
        self.consume(';')
        return ExprStmt(expr)

    def is_type_spec_start(self) -> bool:
        token = self.peek()
        if token is None:
            return False
        # Check if token is a type name
        if token.type == 'IDENT':
            # Type names start with uppercase letter for builtins: Integer, Double, Str, Array, Map, Error, None
            # Accept if first char uppercase
            if token.value[0].isupper():
                return True
        return False

    def parse_import_stmt(self) -> ImportStmt:
        self.consume('retrieve')
        names = self.parse_import_list()
        self.consume('from')
        src_token = self.consume(['IDENT', 'STRING'])
        source = src_token.value
        # Remove quotes if string literal
        if src_token.type == 'STRING':
            source = eval(source)  # safe: contents come from literal
        self.consume(';')
        return ImportStmt(names, source)

    def parse_import_list(self) -> List[str]:
        names = []
        first = self.consume('IDENT')
        names.append(first.value)
        while self.match(','):
            self.consume(',')
            ident = self.consume('IDENT')
            names.append(ident.value)
        return names

    def parse_const_decl(self) -> VarDecl:
        self.consume('const')
        type_spec = self.parse_type_spec()
        name_token = self.consume('IDENT')
        self.consume('=')
        expr = self.parse_expression()
        self.consume(';')
        return VarDecl(type_spec, name_token.value, expr, is_const=True)

    def parse_var_decl(self) -> VarDecl:
        type_spec = self.parse_type_spec()
        name_token = self.consume('IDENT')
        expr: Optional[Node] = None
        if self.match('='):
            self.consume('=')
            expr = self.parse_expression()
        self.consume(';')
        return VarDecl(type_spec, name_token.value, expr, is_const=False)

    def parse_type_spec(self) -> TypeSpec:
        # IDENT ["<" type_spec ("," type_spec)* ">"]
        ident = self.consume('IDENT')
        kind = ident.value
        # Map synonyms for built-in type names to their canonical forms.
        # The grammar uses 'Integer', 'Double', 'Str', etc., but example
        # programs sometimes use 'Int' as a shorthand for 'Integer'.
        synonyms = {
            'Int': 'Integer',
            'String': 'Str'
        }
        kind = synonyms.get(kind, kind)
        args: List[TypeSpec] = []
        if self.match('<'):
            self.consume('<')
            args.append(self.parse_type_spec())
            while self.match(','):
                self.consume(',')
                args.append(self.parse_type_spec())
            self.consume('>')
        return TypeSpec(kind, tuple(args))

    def parse_func_decl(self) -> FuncDecl:
        self.consume('function')
        name_token = self.consume('IDENT')
        self.consume('(')
        params: List[FuncParam] = []
        if not self.match(')'):
            params = self.parse_param_list()
        self.consume(')')
        self.consume('->')
        return_type = self.parse_type_spec()
        body = self.parse_block()
        return FuncDecl(name_token.value, params, return_type, body)

    def parse_param_list(self) -> List[FuncParam]:
        params: List[FuncParam] = []
        while True:
            type_spec = self.parse_type_spec()
            name_token = self.consume('IDENT')
            params.append(FuncParam(type_spec, name_token.value))
            if not self.match(','):
                break
            self.consume(',')
        return params

    def parse_block(self) -> Block:
        self.consume('{')
        statements: List[Node] = []
        while not self.match('}'):
            if self.peek() is None:
                raise ParseError("unterminated block")
            # handle stray semicolons
            if self.match(';'):
                self.consume(';')
                continue
            statements.append(self.parse_statement())
        self.consume('}')
        return Block(statements)

    def parse_if_stmt(self) -> IfStmt:
        self.consume('if')
        self.consume('(')
        condition = self.parse_expression()
        self.consume(')')
        then_block = self.parse_block()
        else_block = None
        if self.match('else'):
            self.consume('else')
            else_block = self.parse_block()
        return IfStmt(condition, then_block, else_block)

    def parse_while_stmt(self) -> WhileStmt:
        self.consume('while')
        self.consume('(')
        condition = self.parse_expression()
        self.consume(')')
        body = self.parse_block()
        return WhileStmt(condition, body)

    def parse_for_stmt(self) -> ForStmt:
        self.consume('for')
        self.consume('(')
        # parse init
        if self.match(';'):
            init = None
            self.consume(';')
        else:
            # try var_decl or expr stmt until ';'
            # look ahead to determine var_decl
            # Save position to backtrack
            saved_pos = self.pos
            try:
                if self.is_type_spec_start():
                    init = self.parse_var_decl()
                else:
                    expr = self.parse_expression()
                    self.consume(';')
                    init = ExprStmt(expr)
            except ParseError:
                # restore and treat as no init
                self.pos = saved_pos
                init = None
                if not self.match(';'):
                    raise
                self.consume(';')
        # parse condition
        if self.match(';'):
            condition = None
            self.consume(';')
        else:
            condition = self.parse_expression()
            self.consume(';')
        # parse post
        if self.match(')'):
            post = None
        else:
            post_expr = self.parse_expression()
            post = post_expr
        self.consume(')')
        body = self.parse_block()
        return ForStmt(init, condition, post, body)

    def parse_attempt_stmt(self) -> AttemptStmt:
        self.consume('attempt')
        try_block = self.parse_block()
        self.consume('fix')
        self.consume('(')
        err_token = self.consume('IDENT')
        self.consume(')')
        catch_block = self.parse_block()
        return AttemptStmt(try_block, err_token.value, catch_block)

    def parse_return_stmt(self) -> ReturnStmt:
        self.consume('return')
        if self.match(';'):
            self.consume(';')
            return ReturnStmt(None)
        value = self.parse_expression()
        self.consume(';')
        return ReturnStmt(value)

    # Expression parsing (Pratt parser)
    def parse_expression(self) -> Node:
        return self.parse_assign()

    # assignment: logic_or ('=' assign)?
    def parse_assign(self) -> Node:
        left = self.parse_logic_or()
        if self.match('='):
            self.consume('=')
            right = self.parse_assign()
            return Assign(left, right)
        return left

    def parse_logic_or(self) -> Node:
        node = self.parse_logic_and()
        while self.match('||'):
            op_token = self.consume('||')
            right = self.parse_logic_and()
            node = BinaryOp(op_token.value, node, right)
        return node

    def parse_logic_and(self) -> Node:
        node = self.parse_equality()
        while self.match('&&'):
            op_token = self.consume('&&')
            right = self.parse_equality()
            node = BinaryOp(op_token.value, node, right)
        return node

    def parse_equality(self) -> Node:
        node = self.parse_comparison()
        while self.match(['==', '!=']):
            op_token = self.consume(['==', '!='])
            right = self.parse_comparison()
            node = BinaryOp(op_token.value, node, right)
        return node

    def parse_comparison(self) -> Node:
        node = self.parse_term()
        while self.match(['<', '>', '<=', '>=']):
            op_token = self.consume(['<', '>', '<=', '>='])
            right = self.parse_term()
            node = BinaryOp(op_token.value, node, right)
        return node

    def parse_term(self) -> Node:
        node = self.parse_factor()
        while self.match(['+', '-']):
            op_token = self.consume(['+', '-'])
            right = self.parse_factor()
            node = BinaryOp(op_token.value, node, right)
        return node

    def parse_factor(self) -> Node:
        node = self.parse_unary()
        while self.match(['*', '/', '%']):
            op_token = self.consume(['*', '/', '%'])
            right = self.parse_unary()
            node = BinaryOp(op_token.value, node, right)
        return node

    def parse_unary(self) -> Node:
        if self.match(['!', '-']):
            op_token = self.consume(['!', '-'])
            operand = self.parse_unary()
            return UnaryOp(op_token.value, operand)
        return self.parse_postfix()

    def parse_postfix(self) -> Node:
        node = self.parse_primary()
        while True:
            if self.match('['):
                self.consume('[')
                index_expr = self.parse_expression()
                self.consume(']')
                node = Index(node, index_expr)
                continue
            if self.match('.'):  # property access
                self.consume('.')
                name_token = self.consume('IDENT')
                node = Member(node, name_token.value)
                continue
            if self.match('('):  # function call
                self.consume('(')
                args: List[Node] = []
                if not self.match(')'):
                    args.append(self.parse_expression())
                    while self.match(','):
                        self.consume(',')
                        args.append(self.parse_expression())
                self.consume(')')
                node = Call(node, args)
                continue
            break
        return node

    def parse_primary(self) -> Node:
        token = self.peek()
        if token is None:
            raise ParseError("unexpected end of input in expression")
        # Literals
        if token.type == 'INT':
            self.consume('INT')
            return Literal(int(token.value), 'Integer')
        if token.type == 'DOUBLE':
            self.consume('DOUBLE')
            return Literal(float(token.value), 'Double')
        if token.type == 'STRING':
            self.consume('STRING')
            # Evaluate Python literal to get string with escapes
            value = eval(token.value)
            return Literal(value, 'Str')
        if token.type == 'CHAR':
            self.consume('CHAR')
            value = eval(token.value)
            return Literal(value, 'Str')
        # Identifier
        if token.type == 'IDENT':
            # If the identifier corresponds to a type name (including generics),
            # parse it as a TypeSpec and wrap as a literal so that type
            # information can be passed to functions like convert_t.
            # Builtin types start with uppercase and include Array/Map with
            # generics. We also treat generics following an uppercase
            # identifier. Otherwise fall back to simple identifier.
            # Save current position to allow lookahead
            name = token.value
            builtin_type_names = {'Integer', 'Double', 'Str', 'Array', 'Map', 'Error', 'None'}
            # Determine if this should be parsed as a TypeSpec
            is_type = False
            if name in builtin_type_names:
                is_type = True
            # Look ahead to detect generic type arguments (e.g., Array<Str>)
            # If next token's value is '<', treat as type spec
            if not is_type and name in {'Array', 'Map'}:
                is_type = True
            # Note: We do NOT treat a following '<' token as indicating a
            # type specification unless the identifier is a known type name
            # (handled above).  This avoids misinterpreting expressions like
            # 'i < 3' as a generic type parameter.
            if is_type:
                # Use parse_type_spec to consume the full type specification
                type_spec = self.parse_type_spec()
                return Literal(type_spec, 'TypeSpec')
            # Otherwise treat as variable identifier
            self.consume('IDENT')
            return Ident(token.value)
        # Array literal
        if token.value == '[':
            self.consume('[')
            elements: List[Node] = []
            if not self.match(']'):
                elements.append(self.parse_expression())
                while self.match(','):
                    self.consume(',')
                    elements.append(self.parse_expression())
            self.consume(']')
            return ArrayLit(elements)
        # Map literal
        if token.value == '{':
            self.consume('{')
            entries: List[Tuple[str, Node]] = []
            if not self.match('}'):
                entries.append(self.parse_map_entry())
                while self.match(','):
                    self.consume(',')
                    entries.append(self.parse_map_entry())
            self.consume('}')
            return MapLit(entries)
        # Grouping
        if token.value == '(':  # grouping parentheses
            self.consume('(')
            expr = self.parse_expression()
            self.consume(')')
            return expr
        raise ParseError(f"unexpected token {token.type} {token.value} at {token.line}:{token.column}")

    def parse_map_entry(self) -> Tuple[str, Node]:
        key_token = self.consume('STRING')
        key_value = eval(key_token.value)
        self.consume(':')
        value = self.parse_expression()
        return (key_value, value)


def parse_program(source: str) -> Program:
    """Parse the given source code into a Program AST using the custom parser."""
    preprocessed = preprocess(source)
    tokens = tokenize(preprocessed)
    parser = Parser(tokens)
    return parser.parse_program()


###############################################################################
# Interpreter implementation
###############################################################################


class FunctionValue:
    """Represents a user-defined Bytie function."""
    def __init__(self, name: str, params: List[FuncParam], return_type: TypeSpec, body: Block, env: Environment):
        self.name = name
        self.params = params
        self.return_type = return_type
        self.body = body
        self.env = env  # closure environment for globals

    def __repr__(self) -> str:
        return f"<function {self.name}>"


class Interpreter:
    """Core interpreter that executes Bytie AST."""
    def __init__(self, debug_level: int = 0, debug_file: str = 'debug.txt'):
        self.global_env = Environment()
        self.modules: Dict[str, Environment] = {}
        self.debug_level = debug_level
        self.debug_fp = open(debug_file, 'w') if debug_level > 0 else None
        # Load standard module
        self.load_standard_module()

    def debug(self, msg: str):
        if self.debug_level > 0:
            if self.debug_fp:
                self.debug_fp.write(msg + '\n')
                self.debug_fp.flush()
            else:
                print(msg)

    # Standard module loading
    def load_standard_module(self):
        std_env = Environment()
        # Built-in functions: print, input, s_to_i, s_to_d, convert_t, convert, error

        def std_print(args: List[Any]) -> Any:
            s = ''.join(to_string(a) for a in args)
            print(s)
            return NoneVal()

        def std_mod(args: List[Any]) -> Any:
           if len(args) != 2:
               raise BytieError(ErrorVal('TypeError', 'mod expects 2 arguments'))
           a, b = args
           if not isinstance(a, int) or not isinstance(b, int):
               raise BytieError(ErrorVal('TypeError', 'mod arguments must be Integer'))
           return a % b

        def std_input(args: List[Any]) -> Any:
            if len(args) != 1:
                raise BytieError(ErrorVal('TypeError', 'input expects 1 argument'))
            prompt = args[0]
            if not isinstance(prompt, str):
                raise BytieError(ErrorVal('TypeError', 'input argument must be Str'))
            try:
                return builtins.input(prompt)
            except EOFError:
                return ''

        def std_s_to_i(args: List[Any]) -> Any:
            if len(args) != 1:
                raise BytieError(ErrorVal('TypeError', 's_to_i expects 1 argument'))
            s = args[0]
            if not isinstance(s, str):
                raise BytieError(ErrorVal('TypeError', 's_to_i expects Str'))
            try:
                return int(s)
            except Exception:
                raise BytieError(ErrorVal('ValueError', f'cannot parse int: {s!r}'))

        def std_s_to_d(args: List[Any]) -> Any:
            if len(args) != 1:
                raise BytieError(ErrorVal('TypeError', 's_to_d expects 1 argument'))
            s = args[0]
            if not isinstance(s, str):
                raise BytieError(ErrorVal('TypeError', 's_to_d expects Str'))
            try:
                return float(s)
            except Exception:
                raise BytieError(ErrorVal('ValueError', f'cannot parse double: {s!r}'))

        def std_convert_t(args: List[Any]) -> Any:
            if len(args) != 3:
                raise BytieError(ErrorVal('TypeError', 'convert_t expects 3 arguments'))
            target_type, source_type, value = args
            # target_type and source_type must be TypeSpec; they are provided as Str and Double? Wait: In user code convert_t(Str, Double, "2.0"); here Str and Double are type names (TypeSpec). Implementation: We'll pass as string names or TypeSpec? parse expression will evaluate type names as Ident referencing TypeSpec? We must treat such parameter evaluation accordingly
            # We'll accept strings representing type names and convert to TypeSpec; or accept TypeSpec directly; we unify
            def normalize_type(t):
                if isinstance(t, TypeSpec):
                    return t
                if isinstance(t, str):
                    # parse simple type name
                    return TypeSpec(t)
                raise BytieError(ErrorVal('TypeError', f'convert_t: invalid type argument {t}'))
            tgt = normalize_type(target_type)
            src = normalize_type(source_type)
            try:
                return convert_value(tgt, src, value)
            except ValueError as e:
                raise BytieError(ErrorVal('ValueError', str(e)))
            except TypeError as e:
                raise BytieError(ErrorVal('TypeError', str(e)))

        def std_convert(args: List[Any]) -> Any:
            # convert(Target, Source, value)
            return std_convert_t(args)

        def std_error(args: List[Any]) -> Any:
            if len(args) != 2:
                raise BytieError(ErrorVal('TypeError', 'error expects 2 arguments'))
            name, message = args
            if not (isinstance(name, str) and isinstance(message, str)):
                raise BytieError(ErrorVal('TypeError', 'error arguments must be Str'))
            return ErrorVal(name, message)

        # Register builtins in std_env
        std_env.values['print'] = BuiltinFunction('print', 1, None, std_print)
        std_env.values['mod'] = BuiltinFunction('mod', 2, TypeSpec.integer(), std_mod)
        std_env.values['input'] = BuiltinFunction('input', 1, None, std_input)
        std_env.values['s_to_i'] = BuiltinFunction('s_to_i', 1, TypeSpec.integer(), std_s_to_i)
        std_env.values['s_to_d'] = BuiltinFunction('s_to_d', 1, TypeSpec.double(), std_s_to_d)
        std_env.values['convert_t'] = BuiltinFunction('convert_t', 3, None, std_convert_t)
        std_env.values['convert'] = BuiltinFunction('convert', 3, None, std_convert)
        std_env.values['error'] = BuiltinFunction('error', 2, TypeSpec.error(), std_error)

        self.modules['Standard'] = std_env
        self.modules['Standard_IO'] = populate_io_environment()

        # Expose selected builtins globally.  Many example programs invoke
        # error() and convert() without explicitly retrieving them from
        # the Standard module.  To support this behaviour, copy these
        # functions into the global environment by default.  Other
        # builtins (e.g., print, input, s_to_i, s_to_d, convert_t) still
        # require an explicit retrieve statement.
        for name in ('error', 'convert'):
            self.global_env.values[name] = std_env.values[name]
            # mark as const so it cannot be overwritten
            self.global_env.consts[name] = True
            # store type information (using Function as a placeholder)
            self.global_env.types[name] = TypeSpec('Function')

    # Public API
    def run(self, program: Program, env: Optional[Environment] = None) -> Any:
        if env is None:
            env = self.global_env
        try:
            return self.execute_block(program.body, env)
        finally:
            if self.debug_fp:
                self.debug_fp.close()

    def execute_block(self, statements: List[Node], env: Environment) -> Any:
        for stmt in statements:
            result = self.execute(stmt, env)
            # propagate return signals
            if isinstance(result, ReturnSignal):
                return result
        return None

    def execute(self, node: Node, env: Environment) -> Any:
        if isinstance(node, VarDecl):
            value = self.evaluate(node.expr, env) if node.expr is not None else None
            env.declare(node.name, node.type_spec, value, node.is_const)
            if self.debug_level >= 2:
                self.debug(f"declare {node.name}: {type_name(env.values[node.name])} = {env.values[node.name]}")
            return None
        if isinstance(node, ImportStmt):
            # Load module
            module_env = self.import_module(node.source, env)
            for name in node.names:
                if name not in module_env.values:
                    raise BytieError(ErrorVal('ImportError', f'module {node.source} has no symbol {name}'))
                env.values[name] = module_env.values[name]
                # propagate constness and types if exists
                if hasattr(module_env, 'consts') and name in module_env.consts:
                    env.consts[name] = module_env.consts[name]
                if hasattr(module_env, 'types') and name in module_env.types:
                    env.types[name] = module_env.types[name]
            return None
        if isinstance(node, FuncDecl):
            func_value = FunctionValue(node.name, node.params, node.return_type, node.body, env)
            env.values[node.name] = func_value
            env.consts[node.name] = True  # functions are const
            env.types[node.name] = TypeSpec('Function')  # placeholder
            if self.debug_level >= 2:
                self.debug(f"define function {node.name}")
            return None
        if isinstance(node, Block):
            # create new environment for block scope (for variables) but share parent for lookups
            block_env = Environment(parent=env)
            res = self.execute_block(node.statements, block_env)
            if isinstance(res, ReturnSignal):
                return res
            return None
        if isinstance(node, IfStmt):
            cond = self.evaluate(node.condition, env)
            truthy = self.is_truthy(cond)
            if self.debug_level >= 3:
                self.debug(f"if condition {cond} -> {truthy}")
            if truthy:
                res = self.execute(node.then_block, env)
                if isinstance(res, ReturnSignal):
                    return res
            elif node.else_block is not None:
                res = self.execute(node.else_block, env)
                if isinstance(res, ReturnSignal):
                    return res
            return None
        if isinstance(node, WhileStmt):
            while True:
                cond = self.evaluate(node.condition, env)
                if not self.is_truthy(cond):
                    break
                res = self.execute(node.body, env)
                if isinstance(res, ReturnSignal):
                    return res
            return None
        if isinstance(node, ForStmt):
            # new scope for init and loop variable declarations
            for_env = Environment(parent=env)
            if node.init is not None:
                # init can be VarDecl or ExprStmt
                if isinstance(node.init, VarDecl):
                    self.execute(node.init, for_env)
                elif isinstance(node.init, ExprStmt):
                    self.evaluate(node.init.expr, for_env)
            while True:
                if node.condition is not None:
                    cond_val = self.evaluate(node.condition, for_env)
                    if not self.is_truthy(cond_val):
                        break
                res = self.execute(node.body, for_env)
                if isinstance(res, ReturnSignal):
                    return res
                if node.post is not None:
                    self.evaluate(node.post, for_env)
            return None
        if isinstance(node, AttemptStmt):
            # Execute try block; catch errors
            try:
                res = self.execute(node.try_block, env)
                # if return in try, propagate
                if isinstance(res, ReturnSignal):
                    return res
            except BytieError as ex:
                # assign error to err_name in a new catch scope
                catch_env = Environment(parent=env)
                # declare err_name as Error type
                catch_env.declare(node.err_name, TypeSpec.error(), ex.err, is_const=False)
                # execute catch block
                res = self.execute(node.catch_block, catch_env)
                if isinstance(res, ReturnSignal):
                    return res
            return None
        if isinstance(node, ReturnStmt):
            value = self.evaluate(node.value, env) if node.value is not None else NoneVal()
            # Type check: we rely on caller to verify return type
            return ReturnSignal(value)
        if isinstance(node, ExprStmt):
            return self.evaluate(node.expr, env)
        # catch any other nodes
        raise NotImplementedError(f"execute: unexpected node type {type(node)}")

    def evaluate(self, node: Node, env: Environment) -> Any:
        # Evaluate expression nodes
        if isinstance(node, Literal):
            return node.value
        if isinstance(node, Ident):
            return env.get(node.name)
        if isinstance(node, ArrayLit):
            # Evaluate elements and create array with dynamic type spec
            # Determine element type from declared variable or from context; here we don't know declared type
            # We'll create array with element type set to Unknown; actual type will be enforced when assigned
            items = [self.evaluate(el, env) for el in node.elements]
            # Determine element type spec: we attempt to unify types of items; default to Str if all are str; else to basic types; we cannot know generics; but we will set to the type of first item
            elem_type: TypeSpec
            if items:
                first_type_name = type_name(items[0])
                # If first is Array<T> or Map<T>, we need to get TypeSpec; parse string representation; but we cannot parse generics; we approximate: if items[0] is ArrayVal, use its elem_type; if MapVal, use its value_type; else base type
                if isinstance(items[0], ArrayVal):
                    elem_type = TypeSpec('Array', (items[0].elem_type,))
                elif isinstance(items[0], MapVal):
                    elem_type = TypeSpec('Map', (items[0].value_type,))
                elif isinstance(items[0], ErrorVal):
                    elem_type = TypeSpec.error()
                elif isinstance(items[0], NoneVal):
                    elem_type = TypeSpec.none()
                elif isinstance(items[0], float):
                    elem_type = TypeSpec.double()
                elif isinstance(items[0], int):
                    elem_type = TypeSpec.integer()
                elif isinstance(items[0], str):
                    elem_type = TypeSpec.string()
                else:
                    elem_type = TypeSpec(type_name(items[0]))
            else:
                # default to Str for empty array; actual type enforced on assignment
                elem_type = TypeSpec.string()
            arr = ArrayVal(elem_type, items)
            return arr
        if isinstance(node, MapLit):
            # Evaluate entries; determine value type spec
            entries: Dict[str, Any] = {}
            value_type: Optional[TypeSpec] = None
            for key, val_node in node.entries:
                val = self.evaluate(val_node, env)
                entries[key] = val
                if value_type is None:
                    # infer type
                    if isinstance(val, ArrayVal):
                        value_type = TypeSpec('Array', (val.elem_type,))
                    elif isinstance(val, MapVal):
                        value_type = TypeSpec('Map', (val.value_type,))
                    elif isinstance(val, ErrorVal):
                        value_type = TypeSpec.error()
                    elif isinstance(val, NoneVal):
                        value_type = TypeSpec.none()
                    elif isinstance(val, float):
                        value_type = TypeSpec.double()
                    elif isinstance(val, int):
                        value_type = TypeSpec.integer()
                    elif isinstance(val, str):
                        value_type = TypeSpec.string()
                    else:
                        value_type = TypeSpec(type_name(val))
            if value_type is None:
                value_type = TypeSpec.string()
            return MapVal(value_type, entries)
        if isinstance(node, UnaryOp):
            operand = self.evaluate(node.operand, env)
            if node.op == '!':
                return 0 if self.is_truthy(operand) else 1
            if node.op == '-':
                if isinstance(operand, bool):
                    operand = int(operand)
                if isinstance(operand, int):
                    return -operand
                if isinstance(operand, float):
                    return -operand
                raise BytieError(ErrorVal('TypeError', f'unary - expects numeric, got {type_name(operand)}'))
            raise BytieError(ErrorVal('TypeError', f'unsupported unary operator {node.op}'))
        if isinstance(node, BinaryOp):
            left = self.evaluate(node.left, env)
            # Short-circuit for && and ||
            if node.op == '&&':
                if not self.is_truthy(left):
                    return 0
                right = self.evaluate(node.right, env)
                return 1 if self.is_truthy(right) else 0
            if node.op == '||':
                if self.is_truthy(left):
                    return 1
                right = self.evaluate(node.right, env)
                return 1 if self.is_truthy(right) else 0
            right = self.evaluate(node.right, env)
            return self.apply_binary_op(node.op, left, right)
        if isinstance(node, Assign):
            # Evaluate right-hand side
            value = self.evaluate(node.value, env)
            # Determine target lvalue
            return self.assign_lvalue(node.target, value, env)
        if isinstance(node, Index):
            target = self.evaluate(node.target, env)
            index = self.evaluate(node.index, env)
            if isinstance(target, ArrayVal):
                # index must be integer
                if not isinstance(index, int):
                    raise BytieError(ErrorVal('TypeError', 'array index must be Integer'))
                # handle negative indexing
                if index < 0:
                    index = len(target.items) + index
                if index < 0 or index >= len(target.items):
                    raise BytieError(ErrorVal('IndexError', f'array index {index} out of range'))
                return target.items[index]
            if isinstance(target, MapVal):
                # key must be string
                if not isinstance(index, str):
                    raise BytieError(ErrorVal('TypeError', 'map key must be Str'))
                if index not in target.entries:
                    raise BytieError(ErrorVal('KeyError', f'key {index!r} not found'))
                return target.entries[index]
            raise BytieError(ErrorVal('TypeError', f'cannot index type {type_name(target)}'))
        if isinstance(node, Member):
            target = self.evaluate(node.target, env)
            if isinstance(target, ErrorVal):
                if node.name == 'name':
                    return target.name
                if node.name == 'message':
                    return target.message
                raise BytieError(ErrorVal('TypeError', f'Unknown error property {node.name}'))
            raise BytieError(ErrorVal('TypeError', f'cannot access property on type {type_name(target)}'))
        if isinstance(node, Call):
            func = self.evaluate(node.func, env)
            args = [self.evaluate(arg, env) for arg in node.args]
            return self.call_function(func, args)
        return None

    def assign_lvalue(self, target: Node, value: Any, env: Environment) -> Any:
        # target can be Ident, Index, or Member (not allowed for now)
        if isinstance(target, Ident):
            env.set(target.name, value)
            return value
        if isinstance(target, Index):
            container = self.evaluate(target.target, env)
            idx = self.evaluate(target.index, env)
            if isinstance(container, ArrayVal):
                if not isinstance(idx, int):
                    raise BytieError(ErrorVal('TypeError', 'array index must be Integer'))
                if idx < 0:
                    idx = len(container.items) + idx
                if idx < 0 or idx >= len(container.items):
                    raise BytieError(ErrorVal('IndexError', f'array index {idx} out of range'))
                # type check value
                try:
                    check_value(value, TypeSpec('Array', (container.elem_type,)))
                except TypeError as e:
                    # For assignment to array element, ensure element type matches container.elem_type
                    try:
                        check_value(value, container.elem_type)
                    except TypeError as ex:
                        raise BytieError(ErrorVal('TypeError', str(ex)))
                container.items[idx] = value
                return value
            if isinstance(container, MapVal):
                if not isinstance(idx, str):
                    raise BytieError(ErrorVal('TypeError', 'map key must be Str'))
                # type check
                try:
                    check_value(value, container.value_type)
                except TypeError as e:
                    raise BytieError(ErrorVal('TypeError', str(e)))
                container.entries[idx] = value
                return value
            raise BytieError(ErrorVal('TypeError', f'cannot assign to index on type {type_name(container)}'))
        # Assignment to member (property) not allowed
        if isinstance(target, Member):
            raise BytieError(ErrorVal('TypeError', 'cannot assign to property'))
        raise BytieError(ErrorVal('TypeError', 'invalid assignment target'))

    def call_function(self, func: Any, args: List[Any]) -> Any:
        if isinstance(func, BuiltinFunction):
            # Check arity; None means variadic
            if func.arity is not None and len(args) != func.arity:
                raise BytieError(ErrorVal('TypeError', f"{func.name} expects {func.arity} arguments"))
            try:
                result = func.fn(args)
                return result
            except BytieError as ex:
                # propagate
                raise ex
        if isinstance(func, FunctionValue):
            # Check argument count
            if len(args) != len(func.params):
                raise BytieError(ErrorVal('TypeError', f"{func.name} expects {len(func.params)} arguments"))
            # Create new environment for call; closure's env is parent
            call_env = Environment(parent=func.env)
            # Bind parameters
            for param, arg in zip(func.params, args):
                # type check
                try:
                    check_value(arg, param.type_spec)
                except TypeError as e:
                    raise BytieError(ErrorVal('TypeError', str(e)))
                call_env.declare(param.name, param.type_spec, arg, is_const=False)
            # Execute body
            try:
                res = self.execute(func.body, call_env)
                if isinstance(res, ReturnSignal):
                    ret_val = res.value
                else:
                    ret_val = NoneVal()
            except ReturnSignal as r:
                ret_val = r.value
            # Check return type
            if func.return_type.kind != 'None':
                # expected return type must match
                try:
                    check_value(ret_val, func.return_type)
                except TypeError as e:
                    raise BytieError(ErrorVal('TypeError', f"return type mismatch in function {func.name}: {e}"))
            return ret_val
        raise BytieError(ErrorVal('TypeError', f'{func} is not callable'))

    def is_truthy(self, value: Any) -> bool:
        # Truthiness rules per specification
        if isinstance(value, bool):
            return value
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

    def apply_binary_op(self, op: str, a: Any, b: Any) -> Any:
        # handle numeric and string operations
        # String concatenation
        if op == '+':
            # If either operand is string, perform concatenation
            if isinstance(a, str) or isinstance(b, str):
                return to_string(a) + to_string(b)
            # numeric addition
            if isinstance(a, float) and isinstance(b, float):
                return a + b
            if isinstance(a, int) and isinstance(b, int):
                return a + b
            # mixed numeric: convert float to int then add
            if isinstance(a, float) and isinstance(b, int):
                return round_to_int_away_from_zero(a) + b
            if isinstance(a, int) and isinstance(b, float):
                return a + round_to_int_away_from_zero(b)
            raise BytieError(ErrorVal('TypeError', f'unsupported + for {type_name(a)} and {type_name(b)}'))
        if op == '-':
            # numeric subtraction
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
                # integer division truncating toward zero
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
            # modulo only for ints
            if isinstance(a, int) and isinstance(b, int):
                if b == 0:
                    raise BytieError(ErrorVal('RuntimeError', 'modulo by zero'))
                return a % b
            raise BytieError(ErrorVal('TypeError', 'modulo requires Integer operands'))
        if op in ('==', '!='):
            eq = self.equal_values(a, b)
            return 1 if (eq if op == '==' else not eq) else 0
        if op in ('<', '>', '<=', '>='):
            # numeric comparisons only
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                # handle mixed numeric
                if isinstance(a, float) and isinstance(b, int):
                    a = round_to_int_away_from_zero(a)
                if isinstance(a, int) and isinstance(b, float):
                    b = round_to_int_away_from_zero(b)
                if op == '<': return 1 if a < b else 0
                if op == '>': return 1 if a > b else 0
                if op == '<=': return 1 if a <= b else 0
                if op == '>=': return 1 if a >= b else 0
            raise BytieError(ErrorVal('TypeError', f'comparison not supported for {type_name(a)} and {type_name(b)}'))
        raise BytieError(ErrorVal('TypeError', f'unknown operator {op}'))

    def equal_values(self, a: Any, b: Any) -> bool:
        # Deep equality per specification
        # Numeric: consider int/double conversions
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
            return all(self.equal_values(x, y) for x, y in zip(a.items, b.items))
        if isinstance(a, MapVal) and isinstance(b, MapVal):
            if a.value_type != b.value_type or len(a.entries) != len(b.entries):
                return False
            for k in a.entries:
                if k not in b.entries:
                    return False
                if not self.equal_values(a.entries[k], b.entries[k]):
                    return False
            return True
        if isinstance(a, ErrorVal) and isinstance(b, ErrorVal):
            return a.name == b.name and a.message == b.message
        if isinstance(a, NoneVal) and isinstance(b, NoneVal):
            return True
        return a == b

    def import_module(self, name: str, current_env: Environment) -> Environment:
        # Name may be without quotes; find file in current directory or relative to program
        if name in self.modules:
            return self.modules[name]
        # If name corresponds to Standard, loaded already
        # Try to load file name.bytie
        module_name = name
        filename = name
        # If module name has no extension, add .bytie
        if not filename.endswith('.bytie'):
            filename = filename + '.bytie'
        # Search relative to current working directory or location of this script
        # Attempt to locate the module file. First check current directory,
        # then check an 'examples' subdirectory if it exists. This helps
        # when running tests from the project root.
        file_path = pathlib.Path(filename)
        if not file_path.exists():
            # Try in examples directory relative to current working directory
            alt = pathlib.Path('examples') / filename
            if alt.exists():
                file_path = alt
            else:
                raise BytieError(ErrorVal('ImportError', f'module {name} not found'))
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        ast_program = parse_program(source)
        module_env = Environment()
        # Provide access to Standard via import if needed
        # When executing module, global scope should be able to import Standard module
        module_interpreter = Interpreter(debug_level=self.debug_level)
        # Copy already loaded Standard module
        module_interpreter.modules.update(self.modules)
        # Execute module AST
        module_interpreter.run(ast_program, module_interpreter.global_env)
        # After run, collect exports: all variables defined in global_env
        module_env.values.update(module_interpreter.global_env.values)
        module_env.consts.update(module_interpreter.global_env.consts)
        module_env.types.update(module_interpreter.global_env.types)
        self.modules[name] = module_env
        return module_env


def run_program(source: str, debug_level: int = 0) -> Any:
    """Convenience function to compile and run a Bytie program from source string."""
    ast_program = parse_program(source)
    interpreter = Interpreter(debug_level=debug_level)
    return interpreter.run(ast_program)


def compile_module(file_path: str, debug_level: int = 0) -> Interpreter:
    """Compile and execute a Bytie file, returning the interpreter instance."""
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    ast_program = parse_program(source)
    interpreter = Interpreter(debug_level=debug_level)
    interpreter.run(ast_program)
    return interpreter