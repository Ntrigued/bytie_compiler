from bytie.interpreter import parse_program, Interpreter


def test_program_27_error_inspect(capsys):
    with open('examples/program_27.bytie', 'r', encoding='utf-8') as f:
        source = f.read()
    ast = parse_program(source)
    interp = Interpreter()
    interp.run(ast)
    out = capsys.readouterr().out.strip()
    assert out == 'err=NotFound :: item missing'