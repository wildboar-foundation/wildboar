import matplotlib.pylab as plt
import numpy as np

from wildboar._utils import check_array_fast
from wildboar.datasets import load_dataset
from wildboar.distance import distance
from wildboar.distance.dtw import dtw_envelop, dtw_lb_keogh, dtw_pairwise_distance

check_array_fast(np.array([1, 2, 3, 2]).reshape(2, 2), ensure_2d=False, allow_nd=True)
x, y = load_dataset("GunPoint")

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
print(dtw_pairwise_distance(x, r=1))
