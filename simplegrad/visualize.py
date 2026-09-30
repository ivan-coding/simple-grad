from pathlib import Path

from graphviz import Digraph

from simplegrad.engine import Value


def collect_graph(root: Value) -> tuple[set[Value], list[tuple[Value, Value]]]:
    nodes, edges = set(), []

    def visit(node: Value):
        if node not in nodes:
            nodes.add(node)
            for child in node._prev:
                edges.append((child, node))
                visit(child)

    visit(root)
    return nodes, edges


def build_graph(root: Value) -> Digraph:
    graph = Digraph(format="png", graph_attr={"rankdir": "LR"})
    nodes, edges = collect_graph(root)

    for node in nodes:
        node_id = str(id(node))
        graph.node(
            name=node_id,
            label=f"{{ {node.label} | data {node.data:.4f} | grad {node.grad:.4f} }}",
            shape="record",
        )
        if node._op:
            operation_id = node_id + node._op
            graph.node(name=operation_id, label=node._op)
            graph.edge(operation_id, node_id)

    for child, parent in edges:
        graph.edge(str(id(child)), str(id(parent)) + parent._op)

    return graph


def render_graph(root: Value, output_path: str | Path) -> Digraph:
    graph = build_graph(root)
    graph.render(output_path, cleanup=True)
    return graph
