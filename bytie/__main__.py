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
from .errors import BytieError
from .ast_json import ast_to_obj, ast_from_obj
from .lower_bytecode import Emitter
from .bytecode import BytecodeVM


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bytie language interpreter")
    parser.add_argument('-v', action='count', default=0, help='increase debug verbosity (can be repeated)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--emit-ast', metavar='BYTIE_FILE', help='emit AST JSON for the given .bytie file')
    group.add_argument('--ast', metavar='AST_JSON_FILE', help='execute AST from a JSON file')
    group.add_argument('--emit-bytecode', metavar='AST_JSON_FILE', help='emit BBC1 bytecode for the given AST JSON file')
    group.add_argument('--bytecode', metavar='BYTECODE_FILE', help='execute BBC1 bytecode file')
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

    # Execute from AST JSON - now compiles to bytecode first
    if args.ast:
        ast_path = Path(args.ast)
        if not ast_path.exists():
            print(f"Error: file {ast_path} not found", file=sys.stderr)
            sys.exit(1)
        with open(ast_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ast_program = ast_from_obj(data)
        
        # Compile AST to bytecode
        emitter = Emitter(ast_program)
        bc_program = emitter.compile()
        
        # Execute bytecode
        vm = BytecodeVM(bc_program, debug_level=args.v)
        try:
            vm.run()
        except BytieError as e:
            print(f"Runtime error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Emit Bytecode mode
    if args.emit_bytecode:
        input_path = Path(args.emit_bytecode)
        if not input_path.exists():
            print(f"Error: file {input_path} not found", file=sys.stderr)
            sys.exit(1)
        
        # Determine if input is AST JSON or source file
        is_ast_json = input_path.name.endswith('.ast.json')
        
        if is_ast_json:
            # Input is already AST JSON, load it directly
            with open(input_path, 'r', encoding='utf-8') as f:
                ast_data = json.load(f)
            ast_obj = ast_from_obj(ast_data)
        else:
            # Input is source file, parse to AST first
            with open(input_path, 'r', encoding='utf-8') as f:
                source = f.read()
            ast_program = parse_program(source)
            ast_obj = ast_program
        
        # Compile AST to bytecode
        emitter = Emitter(ast_obj)
        bc_program = emitter.compile()
        
        # Determine output path (.bytie.bc.json).
        # If the input filename ends with .bytie.ast.json, strip that and add .bytie.bc.json.
        # Otherwise, if it ends with .ast.json, remove that suffix as well. This avoids duplicating
        # the `.bytie` portion when converting files like `program_1.bytie.ast.json`.
        # For source files, replace the source extension with .bytie.bc.json
        name = input_path.name
        out_path: Path
        if name.endswith('.bytie.ast.json'):
            base = name[:-len('.bytie.ast.json')]
            out_name = base + '.bytie.bc.json'
            out_path = input_path.with_name(out_name)
        elif name.endswith('.ast.json'):
            base = name[:-len('.ast.json')]
            out_name = base + '.bytie.bc.json'
            out_path = input_path.with_name(out_name)
        elif name.endswith('.bytie'):
            # Source file with .bytie extension
            base = name[:-len('.bytie')]
            out_name = base + '.bytie.bc.json'
            out_path = input_path.with_name(out_name)
        else:
            # Otherwise replace the last suffix with .bytie.bc.json or append if none
            if input_path.suffix:
                out_path = input_path.with_name(input_path.stem + '.bytie.bc.json')
            else:
                out_path = input_path.with_name(name + '.bytie.bc.json')
        
        with open(out_path, 'w', encoding='utf-8') as out:
            json.dump(bc_program.to_dict(), out, ensure_ascii=False, indent=2)
        # Optionally write disassembly to debug.txt if verbosity
        if args.v > 0:
            vm = BytecodeVM(bc_program, debug_level=args.v)
            # No execution; but we can write disassembly
            # For now, we simply record functions with their code
            with open('debug.txt', 'w', encoding='utf-8') as df:
                df.write(json.dumps(bc_program.to_dict(), indent=2))
        print(str(out_path))
        return

    # Execute bytecode mode
    if args.bytecode:
        bc_path = Path(args.bytecode)
        if not bc_path.exists():
            print(f"Error: file {bc_path} not found", file=sys.stderr)
            sys.exit(1)
        from .bytecode import BytecodeProgram
        with open(bc_path, 'r', encoding='utf-8') as f:
            bc_data = json.load(f)
        program = BytecodeProgram.from_dict(bc_data)
        vm = BytecodeVM(program, debug_level=args.v)
        try:
            vm.run()
        except BytieError as e:
            print(f"Runtime error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Default: execute source file - now compiles to bytecode first
    if not args.program:
        parser.error('missing program file; or use --emit-ast/--ast')
    program_file = Path(args.program)
    if not program_file.exists():
        print(f"Error: file {program_file} not found", file=sys.stderr)
        sys.exit(1)
    with open(program_file, 'r', encoding='utf-8') as f:
        source = f.read()
    ast_program = parse_program(source)
    
    # Compile AST to bytecode
    emitter = Emitter(ast_program)
    bc_program = emitter.compile()
    
    # Execute bytecode
    vm = BytecodeVM(bc_program, debug_level=args.v)
    try:
        vm.run()
    except BytieError as e:
        print(f"Runtime error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()