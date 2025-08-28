"""CLI entry point for the Bytie interpreter.

Usage:
    python -m bytie [-v|-vv|-vvv|-vvvv] <program_file>
    python -m bytie [-v...] --emit-ast <program_file>
    python -m bytie [-v...] --ast <ast_json_file>

Options:
  -v            Increase debug verbosity (can be repeated)
  --emit-ast    Parse the given .bytie file and emit an AST JSON file
  --ast         Execute a previously emitted AST JSON file

Debug information is written to `debug.txt` in the current directory when
verbosity is greater than zero. The interpreter loads standard libraries
automatically and executes the specified Bytie program.
"""

import argparse
import json
import sys
from pathlib import Path
from .interpreter import parse_program, Interpreter
from .ast_json import ast_to_obj, ast_from_obj


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bytie language interpreter")
    parser.add_argument('-v', action='count', default=0, help='increase debug verbosity (can be repeated)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--emit-ast', metavar='BYTIE_FILE', help='emit AST JSON for the given .bytie file')
    group.add_argument('--ast', metavar='AST_JSON_FILE', help='execute AST from a JSON file')
    parser.add_argument('program', nargs='?', help='Bytie program file (.bytie) to execute')
    args = parser.parse_args(argv)

    # Emit AST mode
    if args.emit_ast:
        program_file = Path(args.emit_ast)
        if not program_file.exists():
            print(f"Error: file {program_file} not found", file=sys.stderr)
            sys.exit(1)
        with open(program_file, 'r', encoding='utf-8') as f:
            source = f.read()
        ast_program = parse_program(source)
        obj = ast_to_obj(ast_program)
        out_path = program_file.with_suffix(program_file.suffix + '.ast.json') if program_file.suffix != '' else program_file.with_name(program_file.name + '.ast.json')
        with open(out_path, 'w', encoding='utf-8') as out:
            json.dump(obj, out, ensure_ascii=False, indent=2)
        print(str(out_path))
        return

    # Execute from AST JSON
    if args.ast:
        ast_path = Path(args.ast)
        if not ast_path.exists():
            print(f"Error: file {ast_path} not found", file=sys.stderr)
            sys.exit(1)
        with open(ast_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ast_program = ast_from_obj(data)
        interpreter = Interpreter(debug_level=args.v)
        try:
            interpreter.run(ast_program)
        except Exception as e:
            print(f"Runtime error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Default: execute source file
    if not args.program:
        parser.error('missing program file; or use --emit-ast/--ast')
    program_file = Path(args.program)
    if not program_file.exists():
        print(f"Error: file {program_file} not found", file=sys.stderr)
        sys.exit(1)
    with open(program_file, 'r', encoding='utf-8') as f:
        source = f.read()
    ast_program = parse_program(source)
    interpreter = Interpreter(debug_level=args.v)
    try:
        interpreter.run(ast_program)
    except Exception as e:
        print(f"Runtime error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()