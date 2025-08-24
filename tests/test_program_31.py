import builtins
from bytie.interpreter import parse_program, Interpreter


def test_program_31_repeat(monkeypatch, capsys):
    """Test program 31: user-driven repetition.

    It prompts for a count and echoes a message that many times. We
    simulate user input to supply a count and verify that the output
    matches the expected sequence of lines. The example program uses a
    for-loop starting at 0 and printing "echo k" for each k < times.
    """
    # Provide a test input of 4 to exercise the loop.
    monkeypatch.setattr(builtins, 'input', lambda prompt='': '4')
    with open('examples/program_31.bytie', 'r', encoding='utf-8') as f:
        source = f.read()
    ast = parse_program(source)
    interp = Interpreter()
    interp.run(ast)
    out_lines = capsys.readouterr().out.strip().split('\n')
    # For times=4 we expect four lines, echoing indices 0 through 3.
    assert out_lines == ['echo 0', 'echo 1', 'echo 2', 'echo 3']