import unittest
import numpy as np
import itertools

class TestTree(unittest.TestCase):

    def test_tree(self):
        from pypf.tree import PfTree

        x = np.random.randn(10, 10)
        y = np.random.randint(0, 2, size=10)

        tree = PfTree()
        print(tree.fit(x, y).proba)
