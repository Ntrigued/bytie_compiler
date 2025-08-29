from bytie.interpreter import parse_program, Interpreter


def test_program_5_defines_functions(capsys):
    with open('examples/program_5.bytie', 'r', encoding='utf-8') as f:
        source = f.read()
    ast = parse_program(source)
    interp = Interpreter()
    interp.run(ast)
    out = capsys.readouterr().out.strip()
    # Program 5 only defines functions and imports; no output
    assert out == ''