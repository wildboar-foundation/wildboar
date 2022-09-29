# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause


cdef double scaled_euclidean_distance(
    double *S,
    Py_ssize_t s_length,
    double s_mean,
    double s_std,
    double *T,
    Py_ssize_t t_length,
    double *X_buffer,
    Py_ssize_t *index,
) nogil


cdef double euclidean_distance(
    double *S,
    Py_ssize_t s_length,
    double *T,
    Py_ssize_t t_length,
    Py_ssize_t *index,
) nogil
