__all__ = ["FairNet"]

from .marginalization import *
from .genetic import *
from .viz import *
from .edges import *

import warnings


class FairNet(object):
    def __init__(self, g: object, attrs: dict):
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
                f"{len(self.missing)} missing attribute values detected. Either remove them or replace them with the 'replace_missing_values' method."
            )
        self.enc = self.__label_encoder()

        self.thresh = None
        self.fitness = None
        self.strategy = None
        self.to_remove = None
        self.to_add = None
        self.candidates = []

        self.weights = None
        self.marg_dict = None
        self.disc_nodes = None

        self.logbook = None
        self.solution = None

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
        Computes marginalization scores and detects marginalized nodes given a treshold.

        :param thresh: min marginalization value for a node to be considered as marginalized.
       
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
        fitness,
        strategy,
        to_add=None,
        to_remove=None,
        GA_params=None,
        display=True,
    ):
        """
        The run function is the main function of the GA. It takes in a fitness
        object, a strategy object, and an optional parameter to add edges to the graph.
        It then runs through all of these steps:

        :param self: Reference the object itself
        :param fitness: Define the fitness function
        :param strategy: Select the type of algorithm to use
        :param to_add=None: Add edges to the graph
        :param to_remove=None: Remove edges from the graph
        :param GA_params=None: Pass a dictionary of parameters to the genetic algorithm
        :param display=True: Display the graph of the fitness function
        :param : Store the fitness function
        :return: The graph, the logbook and the individual
        :doc-author: Trelent
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

        self.g = g
        self.logbook = logbook

        self.solution = []
        indexes = [i for i, j in enumerate(individual) if j == 1]
        self.solution = [self.candidates[i][:2] for i in indexes]
        if display:
            plot_GA_eval(logbook=logbook, fitness=self.fitness)

    def marginalization_info(self) -> None:
        """
        marginalization_info 
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

    def replace_missing_values(self, thresh, fitness, GA_params=None, display=True):
        """
        replace_missing_values _summary_

        :param thresh: _description_
        
        :param fitness: _description_
        
        :param GA_params: _description_, defaults to None
        :param display: _description_, defaults to True
        """
        self.thresh = thresh
        self.fitness = fitness.lower()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            attrs, logbook = replace_mv_genetic(self, GA_params=GA_params)

        self.attrs = attrs
        self.missing = []

        if display:
            plot_GA_eval(logbook=logbook, fitness=self.fitness)

    def get_modified_edges(self):
        return self.solution

    def get_graph(self):
        return self.g

    def get_attributes(self):
        return self.attrs

    def is_marginalized(self, node):
        return abs(self.marg_dict[node]) > self.thresh

