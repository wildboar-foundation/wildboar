# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: initializedcheck=False

# Authors: Isak Samsten
# License: BSD 3 clause
import numpy as np
from libc.math cimport log, sqrt, pow, floor
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



cdef double rand_gamma(double alpha, double theta, uint32_t *random_state) noexcept nogil:
    if alpha < 1.0:
        return (
            rand_gamma(alpha + 1, theta, random_state) *
            pow(rand_uniform(0, 1, random_state), 1 / alpha)
        )

    cdef double d = alpha - 1.0 / 3.0
    cdef double c = 1.0 / sqrt(9.0 * d)
    cdef double x, v, u;

    while True:
        x = rand_normal(0, 1, random_state)
        v = 1 + c * x
        if v > 0:
            v = v * v * v
            u = rand_uniform(0, 1, random_state)
            if u < 1.0 - 0.0331 * x * x * x * x:
                return theta * d * v
            if log(u) < 0.5 * x * x + d * (1.0 - v + log(v)):
                return theta * d * v


cdef double rand_beta(double alpha, double beta, uint32_t *random_state) noexcept nogil:
    cdef double x = rand_gamma(alpha, 1, random_state)
    cdef double y = rand_gamma(beta, 1, random_state)
    return x / (x + y)


cdef void shuffle(Py_ssize_t *values, Py_ssize_t length, uint32_t *seed) noexcept nogil:
    cdef Py_ssize_t i, j
    for i in range(length - 1, 0, -1):
        j = rand_int(0, i, seed)
        values[i], values[j] = values[j], values[i]


cdef void rand_uniform_length_interval(
    double min_len,
    double max_len,
    uint32_t *random_seed,
    double *start,
    double *end,
) noexcept nogil:
    cdef double length = rand_uniform(min_len, max_len, random_seed)
    start[0] = rand_uniform(0, 1 - length, random_seed)
    end[0] = start[0] + length


cdef void rand_uniform_start_interval(
    double min_len,
    double max_len,
    uint32_t *random_seed,
    double *start,
    double *end,
) noexcept nogil:
    start[0] = rand_uniform(0, 1 - min_len, random_seed)
    cdef double length = rand_uniform(min_len, min(max_len, 1 - start[0]), random_seed)
    end[0] = start[0] + length


cdef void rand_beta_interval(
    double v, double p, uint32_t *random_seed, double *low, double *high
) noexcept nogil:
    """
    Generate a random interval [low, high] in [0, 1] using beta distributions.

    Parameters
    ----------
    v : double
        Shape parameter for the beta distribution affecting interval width
    p : double
        Probability parameter affecting the number of intervals
    n : Py_ssize_t
        The size parameter controlling interval granularity. Must be
        floor(1/p).
    random_seed : uint32_t*
        Pointer to the random seed for random number generation
    low : double*
        Output parameter for the lower bound of the interval
    high : double*
        Output parameter for the upper bound of the interval

    Notes
    -----
    The function generates intervals using a two-step process:
    1. Generate a random lower bound using Beta(v*z, v*(n-z))
    2. Generate the interval width using Beta(v, v*(n-z-1)) where z is a random
       integer in [0, n).

    E[low-high] ~= p, and each point has p probability of being covered.
    """
    cdef Py_ssize_t n = <Py_ssize_t> floor(1 / p)
    if rand_uniform(0, 1, random_seed) > n * (n + 1) * p - n:
        n += 1

    cdef Py_ssize_t z = rand_int(0, n, random_seed)

    low[0] = 0
    if z != 0:
        low[0] = rand_beta(v * z, v * (n - z), random_seed)

    cdef double w = 1.0
    if z != n - 1:
        w = rand_beta(v, v * (n - z - 1), random_seed)

    high[0] = low[0] + (1 - low[0]) * w
