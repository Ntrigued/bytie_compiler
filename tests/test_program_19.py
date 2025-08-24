from bytie.interpreter import parse_program, Interpreter


def test_program_19_mix(capsys):
    with open('examples/program_19.bytie', 'r', encoding='utf-8') as f:
        source = f.read()
    ast = parse_program(source)
    interp = Interpreter()
    interp.run(ast)
    out = capsys.readouterr().out.strip()
    assert out == 'z = 6'