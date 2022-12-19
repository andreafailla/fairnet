import networkx as nx
from tqdm import tqdm

__all__ = ["get_plausible_edges", "get_removable_edges"]


def get_plausible_edges(fn):
    """
    _summary_

    :param fn: _description_
    :return: _description_
    """
    # pool of edges to add
    plausible = nx.Graph()  # stores plausible links

    for node in tqdm(fn.disc_nodes):
        neighs = set(fn.g.neighbors(node))  # + [node]
        neighs2 = set(nx.ego_graph(fn.g, node, center=False, radius=2).nodes())
        neighs2.difference(neighs)
        for n in neighs2:
            if fn.strategy.endswith("g"):
                if n in fn.disc_nodes:
                    n_neighs = set(fn.g.neighbors(n))
                    weight = len(neighs.intersection(n_neighs))
                    if fn.marg_dict[node] > 0:
                        if fn.attrs[n] != fn.attrs[node]:
                            plausible.add_edge(node, n, weight=weight)
                    elif fn.marg_dict[node] < 0:
                        if fn.attrs[n] == fn.attrs[node]:
                            plausible.add_edge(node, n, weight=weight)
            else:  # local
                n_neighs = set(fn.g.neighbors(n))
                weight = len(neighs.intersection(n_neighs))
                plausible.add_edge(node, n, weight=weight)
    edges = sorted(plausible.edges(data=True), key=lambda t: t[2].get("weight", 1))

    return edges[: round(len(edges) * fn.to_add)]


def get_removable_edges(fn):

    edges = dict()

    for e in fn.g.edges():
        weight = len(set(fn.g.neighbors(e[0])).intersection(set(fn.g.neighbors(e[1]))))
        edges[e] = weight

    removable = dict()

    if fn.strategy.endswith("g"):
        for k, v in edges.items():
            if k[0] in fn.disc_nodes and k[1] in fn.disc_nodes:
                if fn.marg_dict[k[0]] > 0:
                    if fn.attrs[k[0]] == fn.attrs[k[1]]:
                        removable[k] = v
                elif fn.marg_dict[k[0]] < 0:
                    if fn.attrs[k[0]] != fn.attrs[k[1]]:
                        removable[k] = v

    else:  # local
        for k, v in edges.items():
            if k[0] in fn.disc_nodes or k[1] in fn.disc_nodes:
                removable[k] = v

    removable = dict(sorted(removable.items(), key=lambda item: item[1]))

    return list(removable.keys())[: round(len(removable) * fn.to_remove)]
