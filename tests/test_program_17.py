from bytie.interpreter import parse_program, Interpreter


def test_program_17_countdown(capsys):
    with open('examples/program_17.bytie', 'r', encoding='utf-8') as f:
        source = f.read()
    ast = parse_program(source)
    interp = Interpreter()
    interp.run(ast)
    out = capsys.readouterr().out.strip().split('\n')
    assert out == ['t=5', 't=4', 't=3', 't=2', 't=1', 'liftoff!']