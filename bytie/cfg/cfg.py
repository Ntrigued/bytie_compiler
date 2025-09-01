from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Iterable, Any, Optional

from bytie.ast import Node, Program


@dataclass
class CFGNode:
    ast_node: Node
    parents: List["CFGNode"] = field(default_factory=list)
    children: List["CFGNode"] = field(default_factory=list)

    def add_child(self, child: "CFGNode"):
        # Add a directed edge to the child, ensuring no duplicates
        if child not in self.children:
            self.children.append(child)
        # Maintain backreference
        if self not in child.parents:
            child.parents.append(self)

    def add_parent(self, parent: "CFGNode"):
        # Add a directed edge from the parent, ensuring no duplicates
        if parent not in self.parents:
            self.parents.append(parent)
        if self not in parent.children:
            parent.children.append(self)


@dataclass
class CFG:
    nodes: List[CFGNode] = field(default_factory=list)
    start_node: Optional[CFGNode] = None
    end_node: Optional[CFGNode] = None


class CFGBuilder:
    def __init__(self, ast: Program):
        self.ast = ast
        self.cfg = CFG()

    def build(self) -> CFG:
        """Build a *structural* Control Flow Graph (CFG) from the AST.

        Creates one `CFGNode` per AST node and connects each to its immediate
        AST children (including elements of lists/tuples/dicts). This
        representation mirrors the tree shape and does not encode control-flow
        semantics like back-edges or short-circuit evaluation.
        """
        from bytie.ast import Node as _ASTNode  # local alias to avoid shadowing

        node_map: Dict[int, CFGNode] = {}

        def is_ast(x: Any) -> bool:
            return isinstance(x, _ASTNode)

        def mk(n: _ASTNode) -> CFGNode:
            """Return existing CFGNode for AST node `n` or create a new one."""
            k = id(n)
            c = node_map.get(k)
            if c is None:
                c = CFGNode(ast_node=n)
                node_map[k] = c
                self.cfg.nodes.append(c)
            return c

        def iter_children(n: _ASTNode) -> Iterable[_ASTNode]:
            """Yield AST children of `n`, including nested lists/tuples/dicts."""
            for v in vars(n).values():
                if is_ast(v):
                    yield v
                elif isinstance(v, (list, tuple)):
                    for it in v:
                        if is_ast(it):
                            yield it
                elif isinstance(v, dict):
                    for it in v.values():
                        if is_ast(it):
                            yield it
            # primitive types are ignored

        def visit(n: _ASTNode):
            parent = mk(n)
            for ch in iter_children(n):
                child = mk(ch)
                parent.add_child(child)
                visit(ch)

        # Reset nodes list for a fresh build
        self.cfg.nodes.clear()
        if self.ast is None:
            return self.cfg

        visit(self.ast)
        self.cfg.start_node = node_map.get(id(self.ast))
        self.cfg.end_node = self.cfg.nodes[-1] if self.cfg.nodes else None
        return self.cfg


def cfg_to_obj(cfg: CFG) -> Dict[str, Any]:
    """Serialize a CFG to a plain JSON-serializable object.

    The result contains:
    - nodes: list of objects with node index and AST type
    - edges: list of [parent_idx, child_idx] pairs
    - start: index of the start node (or None)
    - end: index of the end node (or None)
    """
    index_map = {id(node): idx for idx, node in enumerate(cfg.nodes)}
    nodes_list = [
        {
            "id": idx,
            "ast_type": node.ast_node.__class__.__name__,
            "node": str(node.ast_node),
        }
        for idx, node in enumerate(cfg.nodes)
    ]
    edges_list: List[List[int]] = []
    for parent_idx, node in enumerate(cfg.nodes):
        for child in node.children:
            edges_list.append({"parent_id": parent_idx, 
                               "child_id": index_map[id(child)],
                               "parent_node": str(node.ast_node),
                               "child_node": str(child.ast_node),
                               })
    return {
        "nodes": nodes_list,
        "edges": edges_list,
        "start": index_map.get(id(cfg.start_node)) if cfg.start_node else None,
        "end": index_map.get(id(cfg.end_node)) if cfg.end_node else None,
    }
