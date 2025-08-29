import builtins
from bytie.interpreter import parse_program, Interpreter


def test_program_3_zero_input(monkeypatch, capsys):
    # Provide input '0' to avoid infinite loop
    monkeypatch.setattr(builtins, 'input', lambda prompt='': '0')
    with open('examples/program_3.bytie', 'r', encoding='utf-8') as f:
        source = f.read()
    ast = parse_program(source)
    interp = Interpreter()
    interp.run(ast)
    out = capsys.readouterr().out.strip()
    assert out == ''