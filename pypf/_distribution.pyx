import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:] get_class_distribution(long[:] idx, long[:] y, int n_classes):
    cdef np.ndarray[np.float64_t] distribution = np.zeros(n_classes)
    cdef double weight = 1.0 / len(idx)
    cdef Py_ssize_t i, j
    cdef Py_ssize_t m = idx.shape[0]
    for i in range(m):
        j = idx[i]
        distribution[y[j]] += weight
    return distribution

