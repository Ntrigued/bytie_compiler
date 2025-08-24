from bytie.interpreter import parse_program, Interpreter


def test_program_10_map_array(capsys):
    with open('examples/program_10.bytie', 'r', encoding='utf-8') as f:
        source = f.read()
    ast = parse_program(source)
    interp = Interpreter()
    interp.run(ast)
    out = capsys.readouterr().out.strip()
    assert out == 'alice first score: 10'