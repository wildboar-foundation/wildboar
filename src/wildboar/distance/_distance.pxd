# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

cimport numpy as np

from wildboar.utils.data cimport Dataset


cdef struct SubsequenceView:
    Py_ssize_t index  # the index of the shapelet sample
    Py_ssize_t start  # the start position
    Py_ssize_t length  # the length of the shapelet
    Py_ssize_t dim  # the dimension of the shapelet
    double mean  # the mean of the shapelet
    double std  # the stanard devision
    void *extra


cdef struct Subsequence:
    Py_ssize_t length
    Py_ssize_t dim
    double mean
    double std
    int ts_index
    int ts_start
    double *data
    void *extra

cdef class SubsequenceDistanceMeasure:

    cdef int reset(self, Dataset dataset) nogil

    cdef int init_transient(
        self,
        Dataset td,
        SubsequenceView *v,
        Py_ssize_t index,
        Py_ssize_t start,
        Py_ssize_t length,
        Py_ssize_t dim,
    ) nogil

    cdef int init_persistent(
        self,
        Dataset dataset,
        SubsequenceView* v,
        Subsequence* s,
    ) nogil

    cdef int free_transient(self, SubsequenceView *t) nogil

    cdef int free_persistent(self, Subsequence *t) nogil

    cdef int from_array(
        self,
        Subsequence *s,
        object obj,
    )

    cdef object to_array(
        self, 
        Subsequence *s
    )

    cdef double transient_distance(
        self,
        SubsequenceView *v,
        Dataset td,
        Py_ssize_t index,
        Py_ssize_t *return_index=*,
    ) nogil

    cdef double persistent_distance(
        self,
        Subsequence *s,
        Dataset td,
        Py_ssize_t index,
        Py_ssize_t *return_index=*,
    ) nogil

    cdef Py_ssize_t transient_matches(
        self,
        SubsequenceView *v,
        Dataset dataset,
        Py_ssize_t index,
        double threshold,
        double **distances,
        Py_ssize_t **indicies,
    ) nogil

    cdef Py_ssize_t persistent_matches(
        self,
        Subsequence *s,
        Dataset dataset,
        Py_ssize_t index,
        double threshold,
        double **distances,
        Py_ssize_t **indicies,
    ) nogil

    cdef double _distance(
        self,
        double *s,
        Py_ssize_t s_len,
        double *x,
        Py_ssize_t x_len,
        Py_ssize_t *return_index=*,
    ) nogil

    cdef Py_ssize_t _matches(
        self,
        double *s,
        Py_ssize_t s_len,
        double *x,
        Py_ssize_t x_len,
        double threshold,
        double **distances,
        Py_ssize_t **indicies,
    ) nogil


cdef class ScaledSubsequenceDistanceMeasure(SubsequenceDistanceMeasure):
    pass


cdef class DistanceMeasure:

    cdef int reset(self, Dataset x, Dataset y) nogil

    cdef double distance(
        self,
        Dataset x,
        Py_ssize_t x_index,
        Dataset y,
        Py_ssize_t y_index,
        Py_ssize_t dim,
    ) nogil

    cdef double _distance(
        self,
        double *x,
        Py_ssize_t x_len,
        double *y,
        Py_ssize_t y_len
    ) nogil
