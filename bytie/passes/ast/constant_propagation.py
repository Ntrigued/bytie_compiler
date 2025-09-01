from bytie.ast import Program
from bytie.passes.ast.abstract_pass import ASTPass
from bytie.ast import *

class ConstantPropagationPass(ASTPass):
    def run(self, ast: Program) -> Program:
        for node in ast.body:
            if isinstance(node, FuncDecl):
                node.body = self.run(node.body)
            if isinstance(node, VarDecl):
                if node.expr is not None:
                    if isinstance(node.expr, Literal):
                        node.expr.value = node.expr.value
                    else:
                        node.expr = node.expr



