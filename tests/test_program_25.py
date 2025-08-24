from bytie.interpreter import parse_program, Interpreter


def test_program_25_for_loop(capsys):
    with open('examples/program_25.bytie', 'r', encoding='utf-8') as f:
        source = f.read()
    ast = parse_program(source)
    interp = Interpreter()
    interp.run(ast)
    out_lines = capsys.readouterr().out.strip().split('\n')
    assert out_lines == ['i=0', 'i=1', 'i=2']