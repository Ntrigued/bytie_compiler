# Bytie Compiler

> **Note**: The initial version of this code was almost completely created by OpenAI Agent Mode.

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
- `examples/` – A collection of example programs (`program_1.bytie`
  through `program_31.bytie`) demonstrating language features.  These
  are the same examples referenced in the specification.
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

The interpreter supports an optional `-v` flag (repeatable up to
four times) which enables debug logging.  Debug output is written
to `debug.txt` in the current working directory.  Higher verbosity
levels produce more detailed traces.  For example:

```sh
python -m bytie -vvv examples/program_7.bytie
```

## Standard Library

The standard library is automatically available via the special
module name `Standard`.  Common built‑ins such as `print`, `input`,
`s_to_i`, `s_to_d`, `convert_t`, `convert` and `error` are provided.
The `error` and `convert` functions are also available without an
explicit `retrieve` statement so that example programs which call
`error("name", "message")` or `convert(...)` directly will work as
written.  Other built‑ins must be imported with `retrieve`:

```byt
retrieve print, s_to_i from Standard;
print("hello");
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

## License

This project is provided for educational purposes and does not carry a
specific license.  Feel free to experiment with the interpreter and
the example programs.