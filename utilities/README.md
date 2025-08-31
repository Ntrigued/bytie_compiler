# Utilities

This directory contains utility scripts for working with the Bytie compiler.

## output_bytecode_code.py

A utility script that extracts and formats bytecode from `.bytie.bc.json` files.

### Usage

```bash
# Print bytecode to console
python output_bytecode_code.py <input_file.bc.json>

# Write bytecode to output file
python output_bytecode_code.py <input_file.bc.json> -o <output_file.txt>
```

### Examples

```bash
# Process a bytecode file and display on console
python output_bytecode_code.py examples/program_1.bytie.bc.json

# Process a bytecode file and save to output file
python output_bytecode_code.py examples/program_1.bytie.bc.json -o bytecode_output.txt
```

### Output Format

The script outputs each bytecode instruction on a new line in the format:
```
INSTRUCTION_NAME arg1 arg2 ...
```

For example, if the JSON contains:
```json
"code": [
  ["SCONST", 0],
  ["LOAD_GLOBAL", 0],
  ["CALL", 1]
]
```

The output will be:
```
SCONST 0
LOAD_GLOBAL 0
CALL 1
```

### Supported File Structures

The script can handle both:
- Direct `code` key in the JSON root
- Nested `code` within the first function in a `functions` array

