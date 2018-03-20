import numpy as np
import timeit
from pypf.tree import get_class_distribution

#print(y)
size = 1000
y = np.random.randint(0, 2, size=size)
idx = np.arange(size)

%timeit get_class_distribution(idx, y, 2)


import pypf._distribution
%timeit pypf._distribution.get_class_distribution(idx, y, 2)

