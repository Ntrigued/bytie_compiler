#!/usr/bin/env python3
"""
Utility script to extract and format bytecode from bytie.bc.json files.
Outputs each instruction on a new line in the format: INSTRUCTION_NAME arg1 arg2 ...
"""

import json
import sys
import argparse


def process_bytecode_file(input_file, output_file=None):
    """
    Process a bytie.bc.json file and output the formatted bytecode.
    
    Args:
        input_file (str): Path to the input .bc.json file
        output_file (str, optional): Path to output file. If None, prints to console.
    """
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Handle different possible structures
        code = None
        
        # Check if there's a direct 'code' key
        if 'code' in data:
            code = data['code']
        # Check if there are functions with code
        elif 'functions' in data and len(data['functions']) > 0:
            # Get the first function's code (usually the main function)
            if 'code' in data['functions'][0]:
                code = data['functions'][0]['code']
        
        if code is None:
            print(f"Error: No 'code' key found in {input_file}", file=sys.stderr)
            return 1
        
        # Format each instruction
        formatted_lines = []
        for instruction in code:
            if isinstance(instruction, list) and len(instruction) > 0:
                # Join instruction name and arguments with spaces
                formatted_line = ' '.join(str(arg) for arg in instruction)
                formatted_lines.append(formatted_line)
        
        # Output the formatted code
        if output_file:
            with open(output_file, 'w') as f:
                for line in formatted_lines:
                    f.write(line + '\n')
            print(f"Bytecode written to {output_file}")
        else:
            for line in formatted_lines:
                print(line)
        
        return 0
        
    except FileNotFoundError:
        print(f"Error: File {input_file} not found", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {input_file}: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error processing {input_file}: {e}", file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Extract and format bytecode from bytie.bc.json files"
    )
    parser.add_argument(
        "input_file",
        help="Input .bc.json file to process"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file to write results to (default: print to console)"
    )
    
    args = parser.parse_args()
    
    # Ensure input file has .bc.json extension
    if not args.input_file.endswith('.bc.json'):
        print("Warning: Input file doesn't end with .bc.json", file=sys.stderr)
    
    exit_code = process_bytecode_file(args.input_file, args.output)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
