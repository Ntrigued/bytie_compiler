from bytie.interpreter import parse_program, Interpreter


def test_program_29_multiplication_table(capsys):
    with open('examples/program_29.bytie', 'r', encoding='utf-8') as f:
        source = f.read()
    ast = parse_program(source)
    interp = Interpreter()
    interp.run(ast)
    out_lines = capsys.readouterr().out.strip().split('\n')
    expected = [
        '1x1=1', '1x2=2', '1x3=3',
        '2x1=2', '2x2=4', '2x3=6',
        '3x1=3', '3x2=6', '3x3=9'
    ]
    assert out_lines == expected