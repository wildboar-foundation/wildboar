# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

cdef enum:
    RAND_R_MAX = 2147483647

cdef struct VoseRand:
    Py_ssize_t n
    Py_ssize_t *alias
    double *prob

cdef void vose_rand_init(VoseRand *vr, Py_ssize_t n) nogil

cdef void vose_rand_free(VoseRand *vr) nogil

cdef void vose_rand_precompute(VoseRand *vr, double *p) nogil

cdef Py_ssize_t vose_rand_int(VoseRand *vr, size_t *seed) nogil

cdef size_t rand_r(size_t *seed) nogil

cdef size_t rand_int(size_t min_val, size_t max_val, size_t *seed) nogil

cdef double rand_uniform(double low, double high, size_t *random_state) nogil

cdef double rand_normal(double mean, double std, size_t *random_state) nogil

cdef void shuffle(Py_ssize_t *values, Py_ssize_t length, size_t *seed) nogil