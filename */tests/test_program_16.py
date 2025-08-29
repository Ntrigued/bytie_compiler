import builtins
from bytie.interpreter import parse_program, Interpreter


def test_program_16_sum(monkeypatch, capsys):
    monkeypatch.setattr(builtins, 'input', lambda prompt='': '5')
    with open('examples/program_16.bytie', 'r', encoding='utf-8') as f:
        source = f.read()
    ast = parse_program(source)
    interp = Interpreter()
    interp.run(ast)
    out = capsys.readouterr().out.strip()
    assert out == 'sum(1..5) = 15'