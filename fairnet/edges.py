import networkx as nx
from tqdm import tqdm

__all__ = ["get_plausible_edges", "get_removable_edges"]


def get_plausible_edges(fn: object) -> list:
    """
    compute the pool of plausible edges to add to the graph
    :param fn: the FairNet object
    :return: the list of plausible edges
    """
    # pool of edges to add
    plausible = nx.Graph()  # stores plausible links

    for node in tqdm(fn.disc_nodes):
        neighs = set(fn.g.neighbors(node))  # + [node]
        neighs2 = set(nx.ego_graph(fn.g, node, center=False, radius=2).nodes())
        neighs2.difference(neighs)
        for n in neighs2:
            if fn.strategy.endswith("g"):
                if fn.is_marginalized(n):
                    n_neighs = set(fn.g.neighbors(n))
                    weight = len(neighs.intersection(n_neighs))
                    if fn.marg_dict[node] > 0 and fn.marg_dict[n] > 0:
                        if fn.attrs[n] != fn.attrs[node]:
                            plausible.add_edge(node, n, weight=weight)
                    elif fn.marg_dict[node] < 0 and fn.marg_dict[n] < 0:
                        if fn.attrs[n] == fn.attrs[node]:
                            plausible.add_edge(node, n, weight=weight)

            else:  # local
                n_neighs = set(fn.g.neighbors(n))
                weight = len(neighs.intersection(n_neighs))
                plausible.add_edge(node, n, weight=weight)
    edges = sorted(plausible.edges(data=True), key=lambda t: t[2].get("weight", 1))

    return edges[: round(len(edges) * fn.to_add)]


def get_removable_edges(fn: object) -> list:
    """
    compute the pool of removable edges
    :param fn: the FairNet object
    :return: the list of removable edges
    """

    edges = dict()

    for e in fn.g.edges():
        weight = len(set(fn.g.neighbors(e[0])).intersection(set(fn.g.neighbors(e[1]))))
        edges[e] = weight

    removable = dict()

    if fn.strategy.endswith("g"):
        for k, v in edges.items():
            if fn.is_marginalized(k[0]) and fn.is_marginalized(k[1]):
                if fn.attrs[k[0]] == fn.attrs[k[1]]:
                    if fn.attrs[k[0]] == fn.attrs[k[1]]:
                        removable[k] = v
                elif fn.marg_dict[k[0]] < 0:
                    if fn.attrs[k[0]] != fn.attrs[k[1]]:
                        removable[k] = v

    else:  # local
        for k, v in edges.items():
            if fn.is_marginalized(k[0]) or fn.is_marginalized(k[1]):
                removable[k] = v

    removable = dict(sorted(removable.items(), key=lambda item: item[1]))

    return list(removable.keys())[: round(len(removable) * fn.to_remove)]
