import numpy as np
from pypf._sliding_distance import sliding_distance

s = np.random.randn(100)
t = np.random.randn(10000)
%timeit sliding_distance(s, t)

