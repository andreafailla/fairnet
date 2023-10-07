from unittest import TestCase
import networkx as nx
import random
from fairnet.classes import FairNet


def get_data(all_attrs=True):
    g = nx.karate_club_graph()
    attrs = nx.get_node_attributes(g, 'club')
    if not all_attrs:
        for n in random.sample(list(g.nodes()), 5):  # remove 5 attributes
            del attrs[n]

    return g, attrs


def get_fitted(all_attrs=True):
    g, attrs = get_data(all_attrs=all_attrs)
    fn = FairNet(g, attrs)
    thresh = 0.3
    fn.fit(thresh)
    return fn


class FairNetTest(TestCase):

    def test_fit(self):
        g, attrs = get_data()

        for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
            fn = FairNet(g, attrs)
            fn.fit(thresh)

            self.assertIsInstance(fn.weights, dict)
            self.assertIsInstance(fn.marg_dict, dict)
            self.assertIsInstance(fn.disc_nodes, list)

            self.assertEqual(len(fn.weights), 2)
            self.assertEqual(len(fn.marg_dict), len(attrs))

    def test_algo(self):
        for fitness in ['marg', 'nodes']:
            for strategy in ['al', 'ag', 'rl', 'rg', 'ab', 'rb']:
                fn = get_fitted()
                fn.run(fitness=fitness,
                       strategy=strategy,
                       to_add=1.0,
                       to_remove=1.0,
                       display=False
                       )
                self.assertGreater(len(fn.solution), 0)
                self.assertIsInstance(fn.solution, list)

        # get modified
        edges = fn.get_modified_edges()
        self.assertIsInstance(edges, list)
        self.assertGreater(len(edges), 0)

        # is marginalized
        self.assertTrue(fn.is_marginalized(0))
        self.assertTrue(fn.is_marginalized(1))
        self.assertFalse(fn.is_marginalized(2))

        # get fair graph
        fair_g = fn.get_fair_graph()
        self.assertIsInstance(fair_g, nx.Graph)
        self.assertEqual(len(fair_g.nodes()), len(fn.g.nodes()))
        self.assertNotEqual(len(fair_g.edges()), len(fn.g.edges()))

    def test_replace_missing_values(self):
        g, attrs = get_data(all_attrs=False)
        fn = FairNet(g, attrs)
        fn.replace_missing_values(thresh=.3, fitness='nodes',
                                  display=False)
        self.assertEqual(len(fn.missing), 0)
        self.assertEqual(len(fn.attrs), len(fn.g.nodes()))
