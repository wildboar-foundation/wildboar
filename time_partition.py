import numpy as np
import timeit
from pypf.tree import partition

#print(y)
size = 1000
d = np.random.randn(size)
y = np.random.randint(0, 2, size=size)
idx = np.arange(size)

partition(idx, d, y, 2)
