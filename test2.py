import numpy as np

from pypf import _tree_builder

y = np.array([0, 0, 1, 1, 0], dtype=np.intp)
i = np.array([0, 1, 4], dtype=np.intp)
_tree_builder.test(i, y, 2)
