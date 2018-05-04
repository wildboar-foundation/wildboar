import numpy as np

from pypf.distance import distance
from pypf.distance import matches

# x = np.random.randn(100, 100000)
# i = np.arange(x.shape[0])
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

from scipy.stats import norm

n_samples = 1000
n_features = 10000
n_classes = 2

rng = np.random.RandomState(41)

delta = 0.5
dt = 1

X = (norm.rvs(
    scale=delta**2 * dt,
    size=n_samples * n_features,
    random_state=rng,
).reshape((n_samples, n_features)))

x = X[0, :]
data = X[1:, :]

print(data.strides)

# d, i = distance(
#     x[0:10],
#     data,
#     dim=0,
#     metric="scaled_euclidean",
#     sample=[10, 100, 22],
#     return_index=True,
# )

# print(d)
# print(i)

d, i = distance(
    x[0:10],
    data,
    dim=0,
    metric="scaled_dtw",
    metric_params={"r": 3},
    sample=None,
    return_index=True,
)

# print(d)
# print(i)

# d, i = matches(
#     x[0:10],
#     data,
#     0.37,
#     dim=0,
#     metric="euclidean",
#     sample=10,
#     return_distance=True,
# )

# print(d)
# print(i)
#print(len(d))
#print(len(i))

# mask = matches([0, 10, 1], x, 2, sample=None, return_distances=True)
# print(mask)

# print(matches([0, 10, 1], x, 2, sample=None, return_distances=True))
