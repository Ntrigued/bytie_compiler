from bytie.interpreter import parse_program, Interpreter


def test_program_12_negative_indexing(capsys):
    with open('examples/program_12.bytie', 'r', encoding='utf-8') as f:
        source = f.read()
    ast = parse_program(source)
    interp = Interpreter()
    interp.run(ast)
    out = capsys.readouterr().out.strip()
    assert out == 'last: e, second_last: d'