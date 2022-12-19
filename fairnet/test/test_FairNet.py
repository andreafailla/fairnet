import unittest
from fairnet.algorithms.sorting import *
from fairnet.classes.FairNet import *
import networkx as nx


def load_network(remove_missing=True):
    g = nx.Graph()

    with open("bt_symmetric.csv") as f:
        for l in f.readlines()[1:]:
            tid, a, b, rssi = l.rstrip().split(",")
            g.add_edge(int(a), int(b), tid=tid)

    attrs = {n: None for n in g.nodes()}  # also fix missing data
    with open("genders.csv") as f:
        for l in f.readlines()[1:]:
            node, gender = l.rstrip().split(",")
            attrs[int(node)] = gender
        nx.set_node_attributes(g, attrs, name="gender")  # probabilmente inutile
    if remove_missing:
        missing = []
        for n in attrs:
            if attrs[n] is None:
                missing.append(n)
        g.remove_nodes_from(missing)

    return g


class TestAlgorithms(unittest.TestCase):
    def test_sort_profiles(self):
        pls = Profiles()
        pls.add_profile(Profile("John", 20, "M"))
        pls.add_profile(Profile("Jane", 25, "F"))
        pls.add_profile(Profile("Jack", 22, "M"))
        pls.add_profile(Profile("Jill", 21, "F"))
        pls.add_profile(Profile("Jenny", 23, "F"))
        pls.add_profile(Profile("Jared", 24, "M"))

        age_sorted = sort_profiles_by_age(pls)
        self.assertEqual(age_sorted.profiles[0].name, "John")

        name_sorted = sort_profiles_by_name(pls)
        self.assertEqual(name_sorted.profiles[0].name, "Jack")


if __name__ == "__main__":
    unittest.main()
