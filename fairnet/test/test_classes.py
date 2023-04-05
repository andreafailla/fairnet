from unittest import TestCase
from fairnet import FairNet


class FairNetTest(TestCase):

    @staticmethod
    def get_data(self, all_attrs=True):
        g = nx.karate_club_graph()
        attrs = nx.get_node_attributes(g, 'club')
        return g, attrs

    @staticmethod
    def get_fitted(self, all_attrs=True):
        g, attrs = self.get_data(all_attrs=all_attrs)
        fn = FairNet(g, attrs)
        thresh = 0.3
        fn.fit(thresh)
        return fn

    def test_fit(self):
        g, attrs = self.get_data()

        for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
            fn = FairNet(g, attrs)
            fn.fit(thresh)

            self.assertIsInstance(fn.weights, dict)
            self.assertIsInstance(fn.marg_dict, dict)
            self.assertIsInstance(fn.disc_nodes, dict)

            self.assertEqual(len(fn.weights), 2)
            self.assertEqual(len(fn.marg_dict), len(attrs))
            self.assertEqual(len(fn.disc_nodes), len(attrs))

    def test_run(self):
        for fitness in ['marg', 'nodes']:
            for strategy in ['al', 'ag', 'rl', 'rg', 'ab', 'rb']:
                fn = self.get_fitted().run(fitness=fitness,
                                           strategy=strategy,
                                           to_add=1.0,
                                           to_remove=1.0,
                                           display=False
                                           )
                self.assertGreater(len(fn.solution), 0)

    def test_replace_missing_values(self):
        self.fail()

    def test_get_modified_edges(self):
        fn = self.get_fitted()
        fn.run(fitness='nodes', strategy='al', to_add=1.0)
        edges = fn.get_modified_edges()
        self.assertIsInstance(edges, list)
        self.assertGreater(len(edges), 0)

    def test_get_graph(self):
        fn = self.get_fitted()

        g = fn.get_graph()

        self.assertIsInstance(g, nx.Graph)

    def test_get_attributes(self):
        fn = self.get_fitted()

        attrs = fn.get_attributes()

        self.assertIsInstance(attrs, dict)

    def test_is_marginalized(self):
        fn = self.get_fitted()
        self.assertTrue(fn.is_marginalized(0))
        self.assertTrue(fn.is_marginalized(1))
        self.assertFalse(fn.is_marginalized(2))

