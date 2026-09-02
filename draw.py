import networkx as nx
import matplotlib.pyplot as plt


def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    nodes, edges = trace(root)

    G = nx.DiGraph()
    labels = {}
    value_node_ids = []
    op_node_ids = []

    for n in nodes:
        uid = str(id(n))
        G.add_node(uid)
        labels[uid] = "%s\ndata %.4f\ngrad %.4f" % (n.label, n.data, n.grad)
        value_node_ids.append(uid)

        if n._op:
            # if this value is a result of some operation, create an op node for it
            op_uid = uid + n._op
            G.add_node(op_uid)
            labels[op_uid] = n._op
            op_node_ids.append(op_uid)
            # and connect this node to it
            G.add_edge(op_uid, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        G.add_edge(str(id(n1)), str(id(n2)) + n2._op)

    pos = nx.spring_layout(G)

    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw_networkx_nodes(G, pos, nodelist=value_node_ids, node_shape='s',
                            node_color='lightblue', node_size=2000, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=op_node_ids, node_shape='o',
                            node_color='lightgray', node_size=800, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

    ax.axis('off')
    fig.tight_layout()
    fig.savefig('graph.png')
    plt.show()

    return fig
