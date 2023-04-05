from collections import Counter
import numpy as np

__all__ = [
    "compute_weights",
    "individual_marginalization_score",
    "compute_marginalization_scores",
    "network_marginalization_score",
    "get_marginalized_nodes",
]


def compute_weights(attrs):
    """
    _summary_

    :param attrs: _description_
    :return: _description_
    """
    weights = dict()
    sizes = dict(Counter(list(attrs.values())))

    for attr, size in sizes.items():
        weights[attr] = 1 - (size - 1) / (len(attrs) - 1)

    return weights


def individual_marginalization_score(g, node, attrs, weights):
    """
    _summary_

    :param g: _description_
    :param node: _description_
    :param attrs: _description_
    :param weights: _description_
    :return: _description_
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


def compute_marginalization_scores(g, attrs, weights):
    """
    _summary_

    :param g: _description_
    :param attrs: _description_
    :param weights: _description_
    :return: _description_
    """

    marg_dict = dict()

    for node in g.nodes():
        marg_dict[node] = individual_marginalization_score(g, node, attrs, weights)

    return marg_dict


def network_marginalization_score(marg_dict):
    """
    _summary_

    :param marg_dict: _description_
    :return: _description_
    """
    return np.mean([abs(v) for v in marg_dict.values()])


def get_marginalized_nodes(marg_dict, threshold):
    """
    _summary_

    :param marg_dict: _description_
    :param threshold: _description_
    :return: _description_
    """
    return [k for k, v in marg_dict.items() if abs(v) > threshold]
