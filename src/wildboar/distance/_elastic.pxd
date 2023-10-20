# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

cdef struct Deque:
    Py_ssize_t *queue
    Py_ssize_t size
    Py_ssize_t capacity
    Py_ssize_t front
    Py_ssize_t back

cdef void deque_init(Deque *c, Py_ssize_t capacity) noexcept nogil

cdef void deque_reset(Deque *c) noexcept nogil

cdef void deque_destroy(Deque *c) noexcept nogil

cdef void deque_push_back(Deque *c, Py_ssize_t v) noexcept nogil

cdef void deque_pop_front(Deque *c) noexcept nogil

cdef void deque_pop_back(Deque *c) noexcept nogil

cdef Py_ssize_t deque_front(Deque *c) noexcept nogil

cdef Py_ssize_t deque_back(Deque *c) noexcept nogil

cdef bint deque_empty(Deque *c) noexcept nogil

cdef Py_ssize_t deque_size(Deque *c) noexcept nogil

cdef void find_min_max(
    const double *T,
    Py_ssize_t length,
    Py_ssize_t r,
    double *lower,
    double *upper,
    Deque *dl,
    Deque *du,
) noexcept nogil

cdef double scaled_dtw_subsequence_distance(
    const double *S,
    Py_ssize_t s_length,
    double s_mean,
    double s_std,
    const double *T,
    Py_ssize_t t_length,
    Py_ssize_t r,
    double *X_buffer,
    double *cost,
    double *cost_prev,
    double *s_lower,
    double *s_upper,
    double *t_lower,
    double *t_upper,
    double *cb,
    double *cb_1,
    double *cb_2,
    Py_ssize_t *index,
) noexcept nogil

cdef double dtw_subsequence_distance(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    Py_ssize_t r,
    double *cost,
    double *cost_prev,
    double *weight_vector,
    Py_ssize_t *index,
) noexcept nogil

cdef double dtw_distance(
    const double *X,
    Py_ssize_t x_length,
    const double *Y,
    Py_ssize_t y_length,
    Py_ssize_t r,
    double *cost,
    double *cost_prev,
    double *weight_vector,
    double min_dist,
) noexcept nogil

cdef double adtw_distance(
    const double *X,
    Py_ssize_t x_length,
    const double *Y,
    Py_ssize_t y_length,
    Py_ssize_t r,
    double *cost,
    double *cost_prev,
    double penalty,
    double min_dist,
) noexcept nogil

cdef double lcss_distance(
    const double *X,
    Py_ssize_t x_length,
    const double *Y,
    Py_ssize_t y_length,
    Py_ssize_t r,
    double threshold,
    double *cost,
    double *cost_prev,
    double *weight_vector,
    double min_dist,
) noexcept nogil

cdef double erp_distance(
    const double *X,
    Py_ssize_t x_length,
    const double *Y,
    Py_ssize_t y_length,
    Py_ssize_t r,
    double g,
    double *gX,
    double *gY,
    double *cost,
    double *cost_prev,
    double min_dist,
) noexcept nogil

cdef double edr_distance(
    const double *X,
    Py_ssize_t x_length,
    const double *Y,
    Py_ssize_t y_length,
    Py_ssize_t r,
    double threshold,
    double *cost,
    double *cost_prev,
    double *weight_vector,
    double min_dist,
) noexcept nogil

cdef double msm_distance(
    const double *X,
    Py_ssize_t x_length,
    const double *Y,
    Py_ssize_t y_length,
    Py_ssize_t r,
    double c,
    double *cost,
    double *cost_prev,
    double *cost_y,
    double min_dist,
) noexcept nogil

cdef double twe_distance(
    const double *X,
    Py_ssize_t x_length,
    const double *Y,
    Py_ssize_t y_length,
    Py_ssize_t r,
    double penalty,
    double stiffness,
    double *cost,
    double *cost_prev,
    double min_dist,
) nogil
