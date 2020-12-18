import math

import numpy as np

from wildboar._utils import check_array_fast
from wildboar.distance import distance
from wildboar.distance.dtw import (
    dtw_pairwise_distance,
    dtw_alignment,
    dtw_distance,
    dtw_envelop,
    dtw_lb_keogh,
)

from wildboar.datasets import load_two_lead_ecg, load_dataset

import matplotlib.pylab as plt

# x, y = load_two_lead_ecg()
# d, i = distance(x[0, 10:20], x, sample=[0, 1, 2, 3, 5, 10], metric='scaled_dtw', metric_params={'r': 0}, return_index=True)
# print(d)
# print(i)
#
# d, i = matches(x[0, 10:20], x, 1.7, sample=[0, 1, 2], metric='scaled_euclidean', metric_params={'r': 0}, return_distance=True)
# print(d)
# print(i)

# print("DIST 1")
# print(distance([1, 2, 3], x, dim=0))
check_array_fast(np.array([1, 2, 3, 2]).reshape(2, 2), ensure_2d=False, allow_nd=True)
x, y = load_dataset("GunPoint")
# rnd = np.random.RandomState(123)
# x = rnd.randn(100)
# y = rnd.randn(100)

l, u = dtw_envelop(x[-1], r=10)
lu = dtw_lb_keogh(x[-1], x[0], r=10)
plt.plot(x[-1])
plt.plot(x[0])
plt.plot(l, "b--")
plt.plot(u, "r--")
plt.plot(lu, "g--")
plt.show()

x = np.array([[1, 10, 2, 3], [1, 2, 3, 4], [2, 1, 2, 2]])
#
print(
    distance(
        [1, 2, 3], x, metric="scaled_dtw", metric_params={"r": 0}, return_index=True
    )
)
# print(np.linalg.norm(s - x))
#
# x = np.random.randn(100, 100)
print(dtw_pairwise_distance(x, r=1))
# t = np.array([-1, -2, -3, -4, 10, 1, 1, 2], dtype=float)
# s, t = t, s

# s = (s - np.mean(s)) / np.std(s)
# t = (t - np.mean(t)) / np.std(t)
# print(s)
# print(t)

# alignment = dtw_alignment(s, t, r=1, out=np.zeros((s.shape[0], t.shape[0])))
# print(np.sqrt(alignment[-1, -1]))
#
# s_scaled = (s - np.mean(s)) / np.std(s)
# t_scaled = (t - np.mean(t)) / np.std(t)

# print(distance(s, t, metric="dtw", metric_params={"r": 1}, subsequence_distance=False))
# print(
#     distance(
#         s, t, metric="scaled_dtw", metric_params={"r": 2}, subsequence_distance=False
#     )
# )
# print(
#     distance(
#         s_scaled,
#         t_scaled,
#         metric="dtw",
#         metric_params={"r": 2},
#         subsequence_distance=False,
#     )
# )
# print(dtw_distance(s, t, r=1))
# print(dtw_distance(s, t, r=2, scale=True))
# print(dtw_distance(s_scaled, t_scaled, r=2, scale=False))
# print(np.sqrt(dtw_alignment(s, t, r=2)[-1, -1]))


# x = np.random.randn(1000000)
# y = np.random.randn(1000000)
# print(distance(x, y, metric="dtw", metric_params={"r": 2}, subsequence_distance=False))
# print(dtw_distance(x, y, r=1))
# import timeit
#
# print(timeit.timeit(lambda: dtw_distance(x, y, r=1), number=10))
# print(
#     timeit.timeit(
#         lambda: distance(
#             x, y, metric="dtw", metric_params={"r": 1}, subsequence_distance=False
#         ),
#         number=10,
#     )
# )
# print(timeit.timeit(lambda: np.linalg.norm(x - y), number=10))

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
