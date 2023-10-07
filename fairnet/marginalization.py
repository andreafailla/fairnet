from collections import Counter

import networkx as nx
import numpy as np

__all__ = [
    "compute_weights",
    "individual_marginalization_score",
    "compute_marginalization_scores",
    "network_marginalization_score",
    "get_marginalized_nodes",
]


def compute_weights(attrs: dict) -> dict:
    """
    Compute the weights of the attributes in the graph. These will be used to compute the marginalization scores.
    :param attrs: the node-to-attribute value dict
    :return: a dictionary containing the weights
    """
    weights = dict()
    sizes = dict(Counter(list(attrs.values())))

    for attr, size in sizes.items():
        weights[attr] = 1 - (size - 1) / (len(attrs) - 1)

    return weights


def individual_marginalization_score(g: nx.Graph, node: int, attrs: dict, weights: dict) -> float:
    """
    Computes the marginalization score of a node.
    :param g: the graph object
    :param node: the node identifier
    :param attrs: the node-to-attribute value dict
    :param weights: the attribute weights
    :return: the marginalization score for the node
    """
    attr = attrs[node]
    marg = 0
    neighs = list(g.neighbors(node))
    neighs_attrs = [attrs[n] for n in neighs]
    size = len(neighs_attrs)

    if size > 2:
        try:
            count = dict(Counter(neighs_attrs))[attr]
            marg = (
                (
                    count
                    * weights[attr]
                    / (count * weights[attr] + (size - count) * (1 - weights[attr]))
                )
                - 0.5
            ) * 2
        except:
            marg = 1
    else:
        marg = 0
    return marg


def compute_marginalization_scores(g: nx.Graph, attrs: dict, weights: dict) -> dict:
    """
    Computes the marginalization scores for all nodes in the graph.

    :param g: the graph object
    :param attrs: the node-to-attribute value dict
    :param weights: the attribute weights
    :return: a dictionary containing the marginalization scores
    """

    marg_dict = dict()

    for node in g.nodes():
        marg_dict[node] = individual_marginalization_score(g, node, attrs, weights)

    return marg_dict


def network_marginalization_score(marg_dict):
    """
    Computes the marginalization score for the entire network. The network marginalization score is the average of the
    absolute values of the marginalization scores of the nodes.
    :param marg_dict:
    :return:
    """
    return np.mean([abs(v) for v in marg_dict.values()])


def get_marginalized_nodes(marg_dict: dict, threshold: float) -> list:
    """
    Returns the nodes whose marginalization score is greater than the threshold.
    :param marg_dict: a dictionary containing the marginalization scores
    :param threshold: the threshold
    :return: a list of nodes
    """
    return [k for k, v in marg_dict.items() if abs(v) > threshold]
