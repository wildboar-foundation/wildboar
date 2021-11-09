import matplotlib.pylab as plt

from wildboar.datasets import load_dataset
from wildboar.distance import (
    paired_distance,
    paired_subsequence_match,
    pairwise_distance,
    pairwise_subsequence_distance,
    subsequence_match,
)
from wildboar.distance.dtw import dtw_envelop, dtw_lb_keogh, dtw_pairwise_distance

x, y = load_dataset("GunPoint")
min_dist, min_ind = pairwise_subsequence_distance(
    x[[2, 3], 0:20],
    x[40:45],
    metric="scaled_dtw",
    metric_params={"r": 1.0},
    return_index=True,
)
print(list(map(list, min_dist)))
print(list(map(list, min_ind)))

ind = subsequence_match(x[0, 20:30].reshape(1, -1), x[0:3], 0.01)
print(ind)

ind = paired_subsequence_match(x[0:3, 20:30], x[0:3], 0.01)
print(ind)

dist = paired_distance(x[0:10], x[10:20])
print(dist)

dist = paired_distance(x[0:3], x[30:33], metric="dtw", metric_params={"r": 1.0})
print(list(dist))

dist = pairwise_distance(x[0:3], x[10:15], metric="dtw", metric_params={"r": 1.0})
print([list(d) for d in dist])
l, u = dtw_envelop(x[-1], r=10)
min_dist, lu = dtw_lb_keogh(x[-1], x[0], r=10)
plt.plot(x[-1])
plt.plot(x[0])
plt.plot(l, "b--")
plt.plot(u, "r--")
plt.plot(lu, "g--")
plt.show()

print(dtw_pairwise_distance(x, r=1))
