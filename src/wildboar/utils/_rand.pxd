# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause
from numpy cimport uint32_t


cdef enum:
    RAND_R_MAX = 2147483647

cdef struct VoseRand:
    Py_ssize_t n
    Py_ssize_t *alias
    double *prob

cdef void vose_rand_init(VoseRand *vr, Py_ssize_t n) noexcept nogil

cdef void vose_rand_free(VoseRand *vr) noexcept nogil

cdef void vose_rand_precompute(VoseRand *vr, const double *p) noexcept nogil

cdef Py_ssize_t vose_rand_int(VoseRand *vr, uint32_t *seed) noexcept nogil

cdef uint32_t rand_r(uint32_t *seed) noexcept nogil

cdef int rand_int(int min_val, int max_val, uint32_t *seed) noexcept nogil

cdef double rand_uniform(double low, double high, uint32_t *random_state) noexcept nogil

cdef double rand_normal(double mean, double std, uint32_t *random_state) noexcept nogil

cdef void shuffle(Py_ssize_t *values, Py_ssize_t length, uint32_t *seed) noexcept nogil

cdef class RandomSampler:

    cdef const double[::1] weights

    cdef Py_ssize_t upper

    cdef VoseRand vr

    cdef Py_ssize_t rand_int(self, uint32_t *seed) noexcept nogil
