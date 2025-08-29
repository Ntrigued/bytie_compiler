from bytie.interpreter import parse_program, Interpreter


def test_program_9_string_and_arith(capsys):
    with open('examples/program_9.bytie', 'r', encoding='utf-8') as f:
        source = f.read()
    ast = parse_program(source)
    interp = Interpreter()
    interp.run(ast)
    out = capsys.readouterr().out.strip()
    # Expected: 7 + round(2.9) = 10
    assert out == 'a(7) + b(2.9) -> 10'