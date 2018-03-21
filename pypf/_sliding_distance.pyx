import numpy as np
cimport numpy as np

cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, INFINITY

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int sliding_distance(double[:] s,
                           double[:, :] X,
                           long[:] idx,
                           double[:] out) nogil except -1:
    cdef Py_ssize_t i, j
    cdef Py_ssize_t m = idx.shape[0]
    cdef Py_ssize_t n = X.shape[1]
    cdef double* buf = <double*>malloc(n * 2 * sizeof(double))
    if not buf:
        return -1
    try:
        for i in range(m):
            j = idx[i]
            out[i] = sliding_distance_(s, X, j, buf)
        return 0
    finally:
        free(buf)


cpdef sliding_distance_one(double[:] s, double[:, :] X, Py_ssize_t i):
    cdef Py_ssize_t n = X.shape[1]
    cdef double* buf = <double*>malloc(n * 2 * sizeof(double))
    if not buf:
        raise MemoryError()
    cdef double dist = sliding_distance_(s, X, i, buf)
    try:
        return dist
    except:
        free(buf)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float sliding_distance_(double[:] s, double[:,:] X, Py_ssize_t ts, double* buf) nogil:
    cdef Py_ssize_t m = s.shape[0]
    cdef Py_ssize_t n = X.shape[1]
    cdef double d = 0
    cdef double mean = 0
    cdef double sigma = 0
    cdef double dist = 0
    cdef double min_dist = INFINITY

    cdef double ex = 0
    cdef double ex2 = 0
    cdef Py_ssize_t i, j
    for i in range(n):
        d = X[ts, i]
        ex += d
        ex2 += (d * d)
        buf[i % m] = d
        buf[(i % m) + m] = d
        if i >= m - 1:
            j = (i + 1) % m
            mean = ex / m
            sigma = sqrt((ex2 / m) - (mean * mean))
            dist = distance(s, buf, j, m, mean, sigma, min_dist)
            if dist < min_dist:
                min_dist = dist
            ex -= buf[j]
            ex2 -= (buf[j] * buf[j])
    return sqrt(min_dist / m)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double distance(double[:] s,
                     double* buf,
                     Py_ssize_t j,
                     Py_ssize_t m,
                     double mean,
                     double std,
                     double bsf) nogil:
    cdef double sf = 0
    cdef double x = 0
    cdef Py_ssize_t i
    for i in range(m):
        if sf >= bsf:
            break
        if std == 0:
            x = s[i]
        else:
            x = (buf[i + j] - mean) / std - s[i]
        sf += x * x
    return sf
