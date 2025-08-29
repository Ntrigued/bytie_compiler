from bytie.interpreter import parse_program, Interpreter


def test_program_20_deconstructed_multiply(capsys):
    with open('examples/program_20.bytie', 'r', encoding='utf-8') as f:
        source = f.read()
    ast = parse_program(source)
    interp = Interpreter()
    interp.run(ast)
    out = capsys.readouterr().out.strip()
    # The provided implementation yields 0 due to missing loop decrement
    assert out == '7 * -4 = 0'