# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: initializedcheck=False

# Authors: Isak Samsten
# License: BSD 3 clause
import numpy as np
from libc.math cimport log, sqrt
from libc.stdlib cimport free, malloc


cdef inline uint32_t DEFAULT_SEED = 1

cdef class RandomSampler:

    def __cinit__(self, Py_ssize_t upper, const double[::1] weights=None):
        self.weights = weights
        self.upper = upper
        if self.weights is not None:
            if upper != weights.shape[0]:
                raise ValueError("upper != len(weights)")

            vose_rand_init(&self.vr, len(weights))
            vose_rand_precompute(&self.vr, &weights[0])

    def __reduce__(self):
        weights = np.asarray(self.weights) if self.weights is not None else None
        return self.__class__, (self.upper, weights)

    def __dealloc__(self):
        if self.weights is not None:
            vose_rand_free(&self.vr)

    cdef Py_ssize_t rand_int(self, uint32_t *seed) noexcept nogil:
        if self.weights is not None:
            return vose_rand_int(&self.vr, seed)
        else:
            return rand_int(0, self.upper, seed)


# https://jugit.fz-juelich.de/mlz/ransampl/-/blob/master/lib/ransampl.c
cdef void vose_rand_init(VoseRand *vr, Py_ssize_t n) noexcept nogil:
    vr.prob = <double*> malloc(sizeof(double) * n)
    vr.alias = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * n)
    vr.n = n

cdef void vose_rand_free(VoseRand *vr) noexcept nogil:
    free(vr.prob)
    free(vr.alias)

cdef void vose_rand_precompute(VoseRand *vr, const double *p) noexcept nogil:
    cdef Py_ssize_t n = vr.n
    cdef Py_ssize_t i, a, g
    cdef double *P = <double*> malloc(sizeof(double) * n)
    cdef Py_ssize_t *S = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * n)
    cdef Py_ssize_t *L = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * n)

    cdef double s = 0
    for i in range(n):
        s += p[i]

    for i in range(n):
        P[i] = p[i] * n / s

    cdef Py_ssize_t nS = 0
    cdef Py_ssize_t nL = 0
    for i in range(n - 1, -1, -1):
        if P[i] < 1:
            S[nS] = i
            nS += 1
        else:
            L[nL] = i
            nL += 1

    while nS > 0 and nL > 0:
        nS -= 1
        a = S[nS]
        nL -= 1
        g = L[nL]
        vr.prob[a] = P[a]
        vr.alias[a] = g
        P[g] = P[g] + P[a] - 1
        if P[g] < 1:
            S[nS] = g
            nS += 1
        else:
            L[nL] = g
            nL += 1

    while nL > 0:
        nL -= 1
        vr.prob[L[nL]] = 1

    while nS > 0:
        nS -= 1
        vr.prob[S[nS]] = 1

    free(P)
    free(S)
    free(L)


cdef Py_ssize_t vose_rand_int(VoseRand *vr, uint32_t *seed) noexcept nogil:
    cdef double r1 = rand_uniform(0, 1, seed)
    cdef double r2 = rand_uniform(0, 1, seed)
    cdef Py_ssize_t i = <Py_ssize_t> (vr.n * r1)
    if r2 < vr.prob[i]:
        return i
    else:
        return vr.alias[i]

# https://github.com/scikit-learn/scikit-learn/blob/
#  433600e68fbb12e72d8c5e0707916f5603bb7057/sklearn/utils/_random.pxd#L26
# TODO(1.2):
# if (seed[0] == 0): seed[0] = DEFAULT_SEED
#
#    seed[0] ^= <uint32_t>(seed[0] << 13)
#    seed[0] ^= <uint32_t>(seed[0] >> 17)
#    seed[0] ^= <uint32_t>(seed[0] << 5)
#    return seed[0] % ((<uint32_t>RAND_R_MAX) + 1)
cdef inline uint32_t rand_r(uint32_t *seed) noexcept nogil:
    """Returns a pesudo-random number based on the seed."""
    seed[0] = seed[0] * 1103515245 + 12345
    return seed[0] % (<uint32_t> RAND_R_MAX + 1)


cdef int rand_int(int min_val, int max_val, uint32_t *seed) noexcept nogil:
    """Returns a pseudo-random number in the range [`min_val` `max_val`["""
    if min_val == max_val:
        return min_val
    else:
        return min_val + rand_r(seed) % (max_val - min_val)


cdef double rand_uniform(
    double low, double high, uint32_t *random_state
) noexcept nogil:
    """Generate a random double in the range [`low` `high`[."""
    return ((high - low) * <double> rand_r(random_state) / <double> RAND_R_MAX) + low


cdef double rand_normal(double mu, double sigma, uint32_t *random_state) noexcept nogil:
    cdef double x1, x2, w, _y1
    x1 = 2.0 * rand_uniform(0, 1, random_state) - 1.0
    x2 = 2.0 * rand_uniform(0, 1, random_state) - 1.0
    w = x1 * x1 + x2 * x2
    while w >= 1.0:
        x1 = 2.0 * rand_uniform(0, 1, random_state) - 1.0
        x2 = 2.0 * rand_uniform(0, 1, random_state) - 1.0
        w = x1 * x1 + x2 * x2

    w = sqrt((-2.0 * log(w)) / w)
    _y1 = x1 * w
    y2 = x2 * w
    return mu + _y1 * sigma


cdef void shuffle(Py_ssize_t *values, Py_ssize_t length, uint32_t *seed) noexcept nogil:
    cdef Py_ssize_t i, j
    for i in range(length - 1, 0, -1):
        j = rand_int(0, i, seed)
        values[i], values[j] = values[j], values[i]
