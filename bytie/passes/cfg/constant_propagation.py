from ast import Set
from typing import Dict
from bytie.cfg import CFG, CFGNode
from bytie.passes.cfg.abstract_cfg_pass import AbstractCFGPass
from bytie.ast import *

class ConstantPropagationPass(AbstractCFGPass):

    def run(self, cfg: CFG) -> CFG:
        self._run(cfg.nodes[0])
        return cfg

    def _run(self, cfg_node: CFGNode, const_vars: Dict[str, str] = {}) -> Dict[str, str]:
        ast_node: Node = cfg_node.ast_node
        if isinstance(ast_node, Program):
            for node in ast_node.body:
                cfg_node = self.run(CFGNode(ast_node=node), const_vars)
        if isinstance(ast_node, Block):
            block: Block = ast_node
            for node in block.body:
                cfg_node = self.run(CFGNode(ast_node=node), const_vars)
        if isinstance(ast_node, VarDecl):
            var_decl: VarDecl = ast_node
            if var_decl.expr is not None:
                if isinstance(var_decl.expr, Literal):
                    const_vars.add(ast_node.name)
        if isinstance(ast_node, ExprStmt):
            if isinstance(ast_node.expr, Assign):
                assgn: Assign = ast_node.expr
                if isinstance(assgn.target, Ident):
                    ident: Ident = assgn.target
                    if isinstance(assgn.value, Literal):
                       const_vars[ident.name] = assgn.value

                    if isinstance(assgn.value, Ident):
                        if assgn.value.name in const_vars:
                            assgn.value = const_vars[assgn.value.name]
                            const_vars[ident.name] = assgn.value
                        else:
                            # no longer guaranteed to be constant
                            del const_vars[ident.name]
        if isinstance(ast_node, BinaryOp):
            bin_op: BinaryOp = ast_node
            left_op = bin_op.left
            right_op = bin_op.right
            right_const, left_const = False, False
            if isinstance(left_op, Ident) and left_op.name in const_vars:
                bin_op.left = const_vars[left_op.name]
                left_const = True
            if isinstance(right_op, Ident) and right_op.name in const_vars:
                bin_op.right = const_vars[right_op.name]
                right_const = True
            if left_const and right_const:
                # TODO: Implement constant foloding for binary operations
                pass
        if isinstance(ast_node, IfStmt):
            if_stmt: IfStmt = ast_node
            const_vars_then = self.run(if_stmt.then_block, const_vars)
            if if_stmt.else_block is not None:
                const_vars_else = self.run(if_stmt.else_block, const_vars)
            # Only constant variables that survive both branches 
            # are guaranteed to remain constant
            const_vars = {}
            for var_name in const_vars_then.keys().intersection(const_vars_else.keys()):
                const_vars[var_name] = const_vars_then[var_name] 
        if isinstance(ast_node, WhileStmt):
            whl_stmt: WhileStmt = ast_node
            const_vars = self.run(whl_stmt.body, const_vars)
        if isinstance(ast_node, ForStmt):
            for_stmt: ForStmt = ast_node
            const_vars = self.run(for_stmt.body, const_vars)
        
        return const_vars