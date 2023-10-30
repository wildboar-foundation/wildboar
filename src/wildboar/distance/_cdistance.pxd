# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause
from numpy cimport float64_t, ndarray

from ..utils cimport TSArray
from ..utils._misc cimport List

# Threshold for when standard deviation is considered to be zero.
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
        double *distances,
        Py_ssize_t *indices,
    ) noexcept nogil

    cdef Py_ssize_t persistent_matches(
        self,
        Py_ssize_t metric,
        Subsequence *s,
        TSArray X,
        Py_ssize_t index,
        double threshold,
        double *distances,
        Py_ssize_t *indices,
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
        double *distances,
        Py_ssize_t *indicies,
    ) noexcept nogil

    cdef Py_ssize_t persistent_matches(
        self,
        Subsequence *s,
        TSArray X,
        Py_ssize_t index,
        double threshold,
        double *distances,
        Py_ssize_t *indicies,
    ) noexcept nogil

    cdef void transient_profile(
        self,
        SubsequenceView *s,
        TSArray x,
        Py_ssize_t i,
        double *dp,
    ) noexcept nogil

    cdef void persistent_profile(
        self,
        Subsequence *s,
        TSArray x,
        Py_ssize_t i,
        double *dp,
    ) noexcept nogil

    cdef void _distance_profile(
        self,
        const double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        const double *x,
        Py_ssize_t x_len,
        double *dp,
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
        double *distances,
        Py_ssize_t *indicies,
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

    # Compute the distance between X[i] and Y[j].
    # If possible, the computation should be aborted if
    # the distance >= lower_bound. To disable the upper bound
    # one can pass INFINITY.
    #
    # `lower_bound` has the initial best-so-far distance on entry.
    #  - If distance < lower_bound, `lower_bound` is the actual distance on
    #    exit and True is returned.
    #  - Otherwise, `lower_bound` is unchanged and the return value is
    #    False
    cdef bint eadistance(
        self,
        TSArray X,
        Py_ssize_t x_index,
        TSArray Y,
        Py_ssize_t y_index,
        Py_ssize_t dim,
        double *lower_bound,
    ) noexcept nogil

    cdef bint _eadistance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len,
        double *upper_bound,
    ) noexcept nogil


cdef Py_ssize_t dilated_distance_profile(
    Py_ssize_t stride,
    Py_ssize_t dilation,
    Py_ssize_t padding,
    double *kernel,
    Py_ssize_t k_len,
    const double* x,
    Py_ssize_t x_len,
    Metric metric,
    double *x_buffer,
    double *k_buffer,
    double ea,
    double* out,
) noexcept nogil

cdef Py_ssize_t scaled_dilated_distance_profile(
    Py_ssize_t stride,
    Py_ssize_t dilation,
    Py_ssize_t padding,
    double *kernel,
    Py_ssize_t k_len,
    const double* x,
    Py_ssize_t x_len,
    Metric metric,
    double *x_buffer,
    double *k_buffer,
    double ea,
    double* out,
) noexcept nogil
