from bytie.interpreter import parse_program, Interpreter


def test_program_21_map_of_maps(capsys):
    with open('examples/program_21.bytie', 'r', encoding='utf-8') as f:
        source = f.read()
    ast = parse_program(source)
    interp = Interpreter()
    interp.run(ast)
    out = capsys.readouterr().out.strip()
    assert out == 'row2.c2 = 4'