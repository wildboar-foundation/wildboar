cdef extern from "math.h":
    double sqrt(double m)

import numpy as np
cimport numpy as np

cimport cython
ctypedef np.float64_t FLOAT64_t

@cython.boundscheck(False)
@cython.wraparound(False)
def sliding_distance(np.ndarray[FLOAT64_t, ndim=1] s not None,
                     np.ndarray[FLOAT64_t, ndim=1] ts not None):
    cdef int m = s.shape[0]
    cdef int n = ts.shape[0]
    cdef np.ndarray[FLOAT64_t, ndim=1] t = np.empty(ts.shape[0] * 2)
    cdef double d = 0
    cdef double mean = 0
    cdef double sigma = 0
    cdef double dist = 0
    cdef double min_dist = np.inf
    cdef int j = 0
    cdef double ex = 0
    cdef double ex2 = 0
    for i in range(n):
        d = ts[i]
        ex += d
        ex2 += (d * d)
        t[i % m] = d
        t[(i % m) + m] = d
        if i >= m - 1:
            j = (i + 1) % m
            mean = ex / m
            sigma = sqrt((ex2 / m) - (mean * mean))
            dist = distance(s, t, j, m, mean, sigma, min_dist)
            if dist < min_dist:
                min_dist = dist
            ex -= t[j]
            ex2 -= (t[j] * t[j])
    return sqrt(min_dist / m)

def distance(np.ndarray[FLOAT64_t, ndim=1] s not None,
             np.ndarray[FLOAT64_t, ndim=1] t not None,
             int j,
             int m,
             double mean,
             double std,
             double bsf):
    cdef double sf = 0
    cdef double x = 0
    for i in range(m):
        if sf >= bsf:
            break
        if std == 0:
            x = s[i]
        else:
            x = (t[i + j] - mean) / std - s[i]
        sf += x * x
    return sf
