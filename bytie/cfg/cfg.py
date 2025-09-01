from dataclasses import dataclass
from typing import List

from bytie.ast import *

@dataclass
class CFGNode:
    ast_node: Node
    parents: List["CFGNode"]
    children: List["CFGNode"]

    def add_child(self, child: "CFGNode"):
        self.children.append(child)
        child.parents.append(self)
    
    def add_parent(self, parent: "CFGNode"):
        self.parents.append(parent)
        parent.children.append(self)

@dataclass
class CFG:
    nodes: List[CFGNode]
    start_node: CFGNode
    end_node: CFGNode


class CFGBuilder:
    def __init__(self, ast: Program):
        self.ast = ast
        self.cfg = CFG()

    def build(self) -> CFG:
        """Build the Control Flow Graph (CFG) from the AST."""
        # TODO: Implement
        return self.cfg
