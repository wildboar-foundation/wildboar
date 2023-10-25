# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

cdef double scaled_euclidean_distance(
    const double *S,
    Py_ssize_t s_length,
    double s_mean,
    double s_std,
    const double *T,
    Py_ssize_t t_length,
    double *X_buffer,
    Py_ssize_t *index,
) noexcept nogil

cdef double euclidean_distance(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    double min_dist,
    Py_ssize_t *index,
) noexcept nogil

cdef double normalized_euclidean_distance(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    Py_ssize_t *index,
) noexcept nogil

cdef double manhattan_distance(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    double min_dist,
    Py_ssize_t *index,
) noexcept nogil

cdef Py_ssize_t manhattan_distance_matches(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    double threshold,
    double *distances,
    Py_ssize_t *matches,
) noexcept nogil

cdef double minkowski_distance(
    double p,
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    double min_dist,
    Py_ssize_t *index,
) noexcept nogil

cdef double chebyshev_distance(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    double min_dist,
    Py_ssize_t *index,
) noexcept nogil

cdef double cosine_similarity(
    const double *S,
    Py_ssize_t s_length,
    const double *T,
    Py_ssize_t t_length,
    Py_ssize_t *index,
) nogil

