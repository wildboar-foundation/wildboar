import numpy as np

from wildboar.distance import distance, matches

from wildboar.datasets import load_two_lead_ecg

x, y = load_two_lead_ecg()
d, i = distance(x[0, 10:20], x, sample=[0, 1, 2, 3, 5, 10], metric='scaled_dtw', metric_params={'r': 0}, return_index=True)
print(d)
print(i)

d, i = matches(x[0, 10:20], x, 1.7, sample=[0, 1, 2], metric='scaled_euclidean', metric_params={'r': 0}, return_distance=True)
print(d)
print(i)


# from wildboar.distance import matches

# from scipy.stats import norm

# n_samples = 1000
# n_features = 10000
# n_classes = 2
#
# rng = np.random.RandomState(41)
# np.zeros([10, 10])
#
# delta = 0.5
# dt = 1
#
# X = (norm.rvs(
#     scale=delta**2 * dt,
#     size=n_samples * n_features,
#     random_state=rng,
# ).reshape((n_samples, n_features)))
#
# x = X[0, :]
# data = X[1:, :]
#
# d, i = distance(
#     x[0:10],
#     data,
#     dim=0,
#     metric="scaled_dtw",
#     metric_params={"r": 3},
#     sample=None,
#     return_index=True,
# )
#
# d, i = matches(
#     x[0:10],
#     data,
#     0.37,
#     dim=0,
#     metric="euclidean",
#     sample=10,
#     return_distance=True,
# )
#
# print(d)
# print(i)
#
