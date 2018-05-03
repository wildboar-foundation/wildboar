import numpy as np
import timeit

from pypf import _tree_builder

#x = np.random.randn(100, 100000)
#i = np.arange(x.shape[0])
# print(((X[0, 0:100] - np.mean(X[0, 0:100])) / np.std(X[0, 0:100]))[:10])
# _tree_builder.test(X, i)

data = [
    [0, 0, 1, 10, 1],
    [0, 1, 9, 1, 0],
    [0, 0, 1, 10, 1],
    [1, 9, 1, 0, 0],
    [0, 1, 9, 1, 0],
    [0, 1, 2, 3, 4],
    [1, 2, 3, 0, 0],
    [0, 0, 0, 1, 2],
    [0, 0, -1, 0, 1],
    [1, 2, 3, 0, 1],
]
x = np.array(data, dtype=np.float64).reshape(-1, 2, 5)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

from pypf.distance import distance
from pypf.distance import matches

print(x)
d, i = distance(
    np.array([1, 2, 3]),
    x,
    dim=1,
    metric="scaled_dtw",
    sample=None,
    return_index=True,
)
print(d)
print(i)
#print(i)

#mask = matches([0, 10, 1], x, 2, sample=None, return_distances=True)
#print(mask)

#print(matches([0, 10, 1], x, 2, sample=None, return_distances=True))
