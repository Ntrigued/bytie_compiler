"""CLI entry point for the Bytie interpreter.

Usage:
    python -m bytie [-v|-vv|-vvv|-vvvv] <program_file>

The optional -v flags increase the verbosity of debug output. Debug
information is written to `debug.txt` in the current directory when
verbosity is greater than zero. The interpreter loads standard
libraries automatically and executes the specified Bytie program.
"""

import argparse
import sys
from pathlib import Path
from .interpreter import parse_program, Interpreter


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bytie language interpreter")
    parser.add_argument('-v', action='count', default=0, help='increase debug verbosity (can be repeated)')
    parser.add_argument('program', help='Bytie program file (.bytie) to execute')
    args = parser.parse_args(argv)
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