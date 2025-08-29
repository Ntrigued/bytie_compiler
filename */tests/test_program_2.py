from bytie.interpreter import parse_program, Interpreter


def test_program_2_no_output(capsys):
    with open('examples/program_2.bytie', 'r', encoding='utf-8') as f:
        source = f.read()
    ast = parse_program(source)
    interp = Interpreter()
    interp.run(ast)
    out = capsys.readouterr().out.strip()
    # Program 2 has no print statements so output should be empty
    assert out == ''