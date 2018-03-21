import numpy as np

y = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0]])
y = np.array([1, 10, 20, 1, 1, 1, 0])
print(y)

if y.ndim == 1:
    d, labels = np.unique(y, return_inverse=True)
    print(d[labels])
else:
    _, labels = np.nonzero(y)
    d = np.unique(labels)
    print(d[labels])
