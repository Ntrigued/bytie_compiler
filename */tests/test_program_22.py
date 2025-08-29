from bytie.interpreter import parse_program, Interpreter


def test_program_22_attempt_fix(capsys):
    with open('examples/program_22.bytie', 'r', encoding='utf-8') as f:
        source = f.read()
    ast = parse_program(source)
    interp = Interpreter()
    interp.run(ast)
    out_lines = capsys.readouterr().out.strip().split('\n')
    # Should print error message and result
    assert out_lines[0].startswith('oops: TypeError')
    assert out_lines[1] == 'result: -1'