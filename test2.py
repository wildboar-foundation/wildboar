import numpy as np
import timeit

from pypf import _tree_builder

m = 10000
X = np.random.randn(100, 100000)
i = np.arange(X.shape[0])
# print(((X[0, 0:100] - np.mean(X[0, 0:100])) / np.std(X[0, 0:100]))[:10])
# _tree_builder.test(X, i)

x = [
    [0, 0, 1, 10, 1],
    [0, 0, 1, 10, 1],
    [0, 1, 9, 1, 0],
    [1, 9, 1, 0, 0],
    [0, 1, 9, 1, 0],
    [0, 1, 2, 3, 4],
    [1, 2, 3, 0, 0],
    [0, 0, 0, 1, 2],
    [0, 0, -1, 0, 1],
    [1, 2, 3, 0, 1],
]
x = np.array(x, dtype=np.float64)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

random_state = np.random.RandomState(1233)
order = np.arange(10)
random_state.shuffle(order)

x = x[order, :]
y = y[order]

# print(x)
# print(y)

train = np.loadtxt("synthetic_control_TRAIN")
test = np.loadtxt("synthetic_control_TEST")

y = train[:, 0].astype(np.intp)
y -= 1
x = train[:, 1:].astype(np.float64)

x_test = test[:, 1:].astype(np.float64)
y_test = test[:, 0].astype(np.intp)
y_test -= 1

_tree_builder.test_2(x, y, 6, random_state, n_shapelets=1000)
