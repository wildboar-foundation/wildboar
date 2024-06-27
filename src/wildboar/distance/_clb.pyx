# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: initializedcheck=False

# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np
from libc.math cimport sqrt


cdef double sax_distance(
    sax_t *x,
    sax_t *y,
    double *breakpoints,
    Py_ssize_t m,
    Py_ssize_t n,
) noexcept nogil:
    cdef double dist = 0.0
    cdef double v
    cdef sax_t max_diff, min_diff
    cdef Py_ssize_t i
    for i in range(m):
        max_diff = max(x[i], y[i])
        min_diff = min(x[i], y[i])
        if max_diff - min_diff <= 1:
            continue
        else:
            v = breakpoints[max_diff - 1] - breakpoints[min_diff]
            dist += v * v

    return sqrt(n / m) * sqrt(dist)


def pairwise_sax_distance(sax_t[:, :] X, sax_t[:, :] Y, double[:] breakpoints, Py_ssize_t n):
    cdef double[:, :] dist = np.empty((X.shape[0], Y.shape[0]), dtype=float)
    cdef Py_ssize_t i, j
    cdef Py_ssize_t m = X.shape[1]

    with nogil:
        for i in range(0, X.shape[0]):
            for j in range(0, Y.shape[0]):
                dist[i, j] = sax_distance(&X[i, 0], &Y[j, 0], &breakpoints[0], m, n)

    return dist.base
