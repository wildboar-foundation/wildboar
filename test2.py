import numpy as np
import timeit

from pypf import _tree_builder

m = 10000
X = np.random.randn(100, 100000)
i = np.arange(X.shape[0])
print(((X[0, 0:100] - np.mean(X[0, 0:100])) / np.std(X[0, 0:100]))[:10])
_tree_builder.test(X, i)
