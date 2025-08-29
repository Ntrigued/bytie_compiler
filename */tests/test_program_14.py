from bytie.interpreter import parse_program, Interpreter


def test_program_14_safe_to_int(capsys):
    with open('examples/program_14.bytie', 'r', encoding='utf-8') as f:
        source = f.read()
    ast = parse_program(source)
    interp = Interpreter()
    interp.run(ast)
    out = capsys.readouterr().out.strip().split('\n')
    # Expect two lines: error message and summary
    assert out[0].startswith('convert failed: ValueError ->')
    assert out[1] == 'a=123, b=0'