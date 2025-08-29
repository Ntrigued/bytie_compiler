# Bytie Compiler

> **Note**: The initial version of this code was almost completely created through a series of prompts asking ChatGPT to iteratively build:
> -  a grammar, based on a set of around 5 provided example programs (hand-written by a human)
> - a larger set of around 25 example programs based on the hand-written sample and grammar
> - a specification for a virtual machine that could run the entire 30 programs.
> culminating in a prompt to OpenAI Agent Mode to create the compiler in one-shot based on the grammar, 30 example programs, and VM specification.
>
> The compiler that the agent returned was almost completely functional, but I did re-arrange some things, added the post_creation/ directory, and the std/io directories before the first commit.

# Bytie Language Interpreter

This repository contains an implementation of the **Bytie** programming
language.  Bytie is a simple, statically typed language with support
for integers, doubles, strings, arrays, maps, user‑defined functions,
control flow (`if`, `while`, `for`), and structured error handling
(`attempt`/`fix`).  The accompanying virtual machine and standard
library are implemented entirely in Python.

## Project Layout

- `bytie/` – The Python package implementing the parser, interpreter,
  type system and built‑ins.  The entry point is in
  `bytie/__main__.py`, which makes the package runnable with
  `python -m bytie`.
- `bytie/std/` – Standard library modules including:
  - `bytie/std/io/` – File I/O operations module
- `examples/` – A collection of example programs (`program_1.bytie`
  through `program_31.bytie`) demonstrating language features.  These
  are the same examples referenced in the specification.
- `post_creation/` – Additional example programs created after the initial
  implementation, including file I/O examples.
- `tests/` – A comprehensive test suite using `pytest` which
  exercises the interpreter and all example programs.  Each program
  has its own test file along with additional subsystem tests.
- `README.md` – This document.
- `requirements.txt` – Python dependencies needed to run the tests.

## Running Bytie Programs

To execute a Bytie program, invoke the package as a module.  For
example, to run `examples/program_1.bytie` use:

```sh
python -m bytie examples/program_1.bytie
```

The interpreter supports several command-line options:

### Verbosity and Debug Options
- `-v` flag (repeatable up to four times) which enables debug logging.  Debug output is written
to `debug.txt` in the current working directory.  Higher verbosity
levels produce more detailed traces.  For example:

```sh
python -m bytie -vvv examples/program_7.bytie
```

### AST Generation and Execution
- `--emit-ast <program_file>` - Parse a .bytie file and generate a file containing a JSON representation of the abstract syntax tree. The output file will have the same name as the input with `.ast.json` appended.

```sh
python -m bytie --emit-ast examples/program_1.bytie
# Creates examples/program_1.bytie.ast.json
```

- `--ast <ast_json_file>` - Execute a previously generated AST JSON file instead of parsing source code.
```sh
python -m bytie --ast examples/program_1.bytie.ast.json
```

## Standard Library

The standard library is automatically available via two special module names:

### Standard Module
The `Standard` module provides common built‑ins such as `print`, `input`,
`s_to_i`, `s_to_d`, `convert_t`, `convert`, `mod`, and `error`.  The `error` 
and `convert` functions are also available without an explicit `retrieve` statement 
so that example programs which call `error("name", "message")` or `convert(...)` 
directly will work as written.  Other built‑ins must be imported with `retrieve`:

```byt
retrieve print, s_to_i from Standard;
print("hello");
```

### Standard.IO Module
The `Standard_IO` module provides comprehensive file I/O operations:

- **File Management**: `open_file(filename, mode)`, `close_file(fileno)`, `delete_file(filename)`
- **File Operations**: `read_file(fileno)`, `write_file(fileno, data)`
- **File Manipulation**: `rename_file(old_name, new_name)`, `copy_file(source, dest)`, `move_file(source, dest)`
- **File Information**: `file_exists(filename)`

Example usage:
```byt
retrieve open_file, read_file, write_file, close_file from Standard_IO;
retrieve print from Standard;

Integer file_no = open_file('test.txt', 'w');
write_file(file_no, "Hello, World!");
close_file(file_no);
```

## Testing

The `tests/` directory contains a suite of tests based on
`pytest`.  To run the tests locally (assuming you have `pytest`
installed), execute:

```sh
python -m pytest -q
```

The test suite covers all aspects of the language and virtual
machine, including the correct behaviour of each example program.

## Debug Mode

The interpreter includes a simple debug logger.  When run with one or
more `-v` flags, the interpreter will log its execution to a file
named `debug.txt`.  Increasing the number of `v`s increases the
verbosity of the log.  Debug logs include information such as
variable declarations, control flow decisions and function calls.  The
log can be useful when debugging your own Bytie programs.
