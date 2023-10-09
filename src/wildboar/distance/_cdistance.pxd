# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause
from numpy cimport float64_t, ndarray

from ..utils cimport TSArray
from ..utils._misc cimport List


cdef double EPSILON = 1e-13

cdef class MetricList(List):

    cdef int reset(self, Py_ssize_t metric, TSArray X, TSArray Y) noexcept nogil

    cdef double distance(
        self, 
        Py_ssize_t metric, 
        TSArray X,
        Py_ssize_t x_index,
        TSArray Y,
        Py_ssize_t y_index,
        Py_ssize_t dim,
    ) noexcept nogil

    cdef double _distance(
        self,
        Py_ssize_t metric,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len
    ) noexcept nogil

cdef class SubsequenceMetricList(List):

    cdef int reset(self, Py_ssize_t metric, TSArray X) noexcept nogil

    cdef int init_transient(
        self,
        Py_ssize_t metric,
        TSArray X,
        SubsequenceView *v,
        Py_ssize_t index,
        Py_ssize_t start,
        Py_ssize_t length,
        Py_ssize_t dim,
    ) noexcept nogil

    cdef int init_persistent(
        self,
        Py_ssize_t metric, 
        TSArray X,
        SubsequenceView* v,
        Subsequence* s,
    ) noexcept nogil

    cdef int free_transient(self, Py_ssize_t metric, SubsequenceView *t) noexcept nogil

    cdef int free_persistent(self, Py_ssize_t metric, Subsequence *t) noexcept nogil

    cdef int from_array(
        self,
        Py_ssize_t metric, 
        Subsequence *s,
        object obj,
    )

    cdef object to_array(
        self, 
        Py_ssize_t metric, 
        Subsequence *s
    )

    cdef double transient_distance(
        self,
        Py_ssize_t metric, 
        SubsequenceView *v,
        TSArray X,
        Py_ssize_t index,
        Py_ssize_t *return_index,
    ) noexcept nogil

    cdef double persistent_distance(
        self,
        Py_ssize_t metric, 
        Subsequence *s,
        TSArray X,
        Py_ssize_t index,
        Py_ssize_t *return_index,
    ) noexcept nogil

    cdef Py_ssize_t transient_matches(
        self,
        Py_ssize_t metric, 
        SubsequenceView *v,
        TSArray X,
        Py_ssize_t index,
        double threshold,
        double **distances,
        Py_ssize_t **indices,
    ) noexcept nogil

    cdef Py_ssize_t persistent_matches(
        self,
        Py_ssize_t metric, 
        Subsequence *s,
        TSArray X,
        Py_ssize_t index,
        double threshold,
        double **distances,
        Py_ssize_t **indices,
    ) noexcept nogil


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

cdef class SubsequenceMetric:

    cdef int reset(self, TSArray X) noexcept nogil

    cdef int init_transient(
        self,
        TSArray X,
        SubsequenceView *v,
        Py_ssize_t index,
        Py_ssize_t start,
        Py_ssize_t length,
        Py_ssize_t dim,
    ) noexcept nogil

    cdef int init_persistent(
        self,
        TSArray X,
        SubsequenceView* v,
        Subsequence* s,
    ) noexcept nogil

    cdef int free_transient(self, SubsequenceView *t) noexcept nogil

    cdef int free_persistent(self, Subsequence *t) noexcept nogil

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
        TSArray X,
        Py_ssize_t index,
        Py_ssize_t *return_index=*,
    ) noexcept nogil

    cdef double persistent_distance(
        self,
        Subsequence *s,
        TSArray X,
        Py_ssize_t index,
        Py_ssize_t *return_index=*,
    ) noexcept nogil

    cdef Py_ssize_t transient_matches(
        self,
        SubsequenceView *v,
        TSArray X,
        Py_ssize_t index,
        double threshold,
        double **distances,
        Py_ssize_t **indicies,
    ) noexcept nogil

    cdef Py_ssize_t persistent_matches(
        self,
        Subsequence *s,
        TSArray X,
        Py_ssize_t index,
        double threshold,
        double **distances,
        Py_ssize_t **indicies,
    ) noexcept nogil

    cdef double _distance(
        self,
        const double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        const double *x,
        Py_ssize_t x_len,
        Py_ssize_t *return_index=*,
    ) noexcept nogil

    cdef Py_ssize_t _matches(
        self,
        const double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        const double *x,
        Py_ssize_t x_len,
        double threshold,
        double **distances,
        Py_ssize_t **indicies,
    ) noexcept nogil


cdef class ScaledSubsequenceMetric(SubsequenceMetric):
    pass


cdef class Metric:

    cdef int reset(self, TSArray X, TSArray Y) noexcept nogil

    cdef double distance(
        self,
        TSArray X,
        Py_ssize_t x_index,
        TSArray Y,
        Py_ssize_t y_index,
        Py_ssize_t dim,
    ) noexcept nogil

    cdef double _distance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len
    ) noexcept nogil