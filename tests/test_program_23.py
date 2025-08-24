from bytie.interpreter import parse_program, Interpreter


def test_program_23_avg3(capsys):
    with open('examples/program_23.bytie', 'r', encoding='utf-8') as f:
        source = f.read()
    ast = parse_program(source)
    interp = Interpreter()
    interp.run(ast)
    out = capsys.readouterr().out.strip()
    assert out == 'avg3(3,4,5) = 4'