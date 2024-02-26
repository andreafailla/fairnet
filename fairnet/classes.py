__all__ = ["FairNet"]

import networkx as nx

from .marginalization import *
from .genetic import *
from .viz import *
from .edges import *

import warnings


class FairNet(object):
    def __init__(self, g: nx.Graph, attrs: dict):
        """
        Initialize the FairNet object.
        Throws a warning if 'attrs' lacks attribute values for any node.

        :param g: The graph
        :param attrs: The node-to-attribute value dict

        """
        self.g = g.copy()
        self.attrs = {k: v for k, v in attrs.items()}

        self.missing = [
            n for n in self.g.nodes() if n not in self.attrs
        ]  # nodes with missing values
        if len(self.missing) > 0:
            warnings.warn(
                f"{len(self.missing)} missing attribute values detected. Either remove them or replace them with the "
                f"'replace_missing_values' method."
            )
        self.enc = self.__label_encoder()

        self.thresh = None  # marginalization threshold
        self.fitness = None  # fitness function
        self.strategy = None  # strategy
        self.to_remove = None  # percentage of edges to remove
        self.to_add = None  # percentage of edges to add
        self.candidates = []  # list of candidate edges

        self.weights = None  # attribute weights
        self.marg_dict = None  # marginalization scores
        self.disc_nodes = None  # marginalized nodes

        self.logbook = None  # GA logbook
        self.solution = None  # GA best solution
        self.fair_g = None  # fair graph

    def __label_encoder(self):
        enc = dict()

        known = []
        for _, v in self.attrs.items():
            if v not in known:
                known.append(v)
            enc[known.index(v)] = v
        return enc

    def fit(self, thresh: float) -> object:
        """
        Fits the FairNet object. Computes the weights, the marginalization scores, and the marginalized nodes.
        :param thresh: the threshold for marginalization
        :return: self
        """

        self.thresh = thresh

        self.weights = compute_weights(self.attrs)
        self.marg_dict = compute_marginalization_scores(
            self.g, self.attrs, self.weights
        )
        self.disc_nodes = get_marginalized_nodes(self.marg_dict, self.thresh)
        return self

    def run(
        self,
        fitness: str,
        strategy: str,
        to_add: float = None,
        to_remove: float = None,
        GA_params: dict = None,
        display: bool = True,
    ):
        """
        Executes the algorithm to reduce marginalization.
        :param fitness: either 'marg' or 'nodes'
        :param strategy:
        :param to_add: the percentage of edges to add among the plausible ones
        :param to_remove: the percentage of edges to remove among the removable ones
        :param GA_params: the dictionary of parameters for the genetic algorithm
        :param display: whether to display the GA evaluation
        :return:
        """

        self.fitness = fitness
        self.strategy = strategy.lower()
        self.to_remove = to_remove
        self.to_add = to_add

        if self.strategy[0] in "ab":
            edges = get_plausible_edges(self)
            self.candidates.extend(edges)

        if self.strategy[0] in "rb":
            if not isinstance(self.to_remove, float):
                raise ValueError("You must set the 'to_remove' parameter")
            edges = get_removable_edges(self)
            self.candidates.extend(edges)
        print(f"Starting GA over {len(self.candidates)} candidates...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g, logbook, individual = reduce_marginalization_genetic(self, GA_params)

        self.fair_g = g
        self.logbook = logbook

        self.solution = []
        indexes = [i for i, j in enumerate(individual) if j == 1]
        self.solution = [self.candidates[i] for i in indexes]
        if display:
            plot_GA_eval(logbook=logbook, fitness=self.fitness)

    def marginalization_info(self) -> Tuple[float, float]:
        """
        Prints marginalization information. In detail, it prints the weights, the number of marginalized nodes,
        the average marginalization score, and plots the marginalization scores.
        return: score and nodes
        """
        print("=" * 10, "STATS", "=" * 10)
        print("\nWeights:", self.weights)

        print(
            f"\nMarginalized nodes:, {len(self.disc_nodes)} ({round(len(self.disc_nodes) * 100 / len(self.g), 2)}%)"
        )

        print(
            "\nAverage Marginalization Score:",
            network_marginalization_score(self.marg_dict),
        )
        plot_marginalization_scores(self.marg_dict)

        plot_marginalization_scores_by_attr(self.attrs, self.marg_dict)
        return (len(self.disc_nodes), network_marginalization_score(self.marg_dict))

    def replace_missing_values(
        self, thresh: float, fitness: str, GA_params=None, display=True
    ):
        """
        Replaces missing values so as to minimize marginalization.
        :param thresh: the threshold for marginalization
        :param fitness: either 'marg' or 'nodes'
        :param GA_params: the dictionary of parameters for the genetic algorithm
        :param display: whether to display the GA evaluation
        :return:
        """
        self.thresh = thresh
        self.fitness = fitness.lower()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            attrs, logbook = replace_missing_values_genetic(self, GA_params=GA_params)

        self.attrs = attrs
        self.logbook = logbook
        self.missing = []

        if display:
            plot_GA_eval(logbook=logbook, fitness=self.fitness)

    def get_modified_edges(self):
        """
        Returns the edges that have been added or removed by the algorithm.
        :return: a list of tuples (u, v, weight)
        """
        return self.solution

    def get_fair_graph(self):
        """
        Returns the fair graph, i.e., the graph with the added/removed edges.
        :return:
        """
        return self.fair_g

    def get_attributes(self):
        """
        Returns the attributes of the nodes.
        :return: a dictionary {node: attribute_value}
        """
        return self.attrs

    def is_marginalized(self, node):
        """
        Returns True if the node is marginalized, False otherwise.
        :param node: the node to check
        :return: True if the node is marginalized, False otherwise
        """
        return abs(self.marg_dict[node]) > self.thresh
