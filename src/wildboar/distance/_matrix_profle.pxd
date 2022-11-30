# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

cdef void _matrix_profile_stmp(
    const double *x,
    Py_ssize_t x_length,
    const double *y,
    Py_ssize_t y_length,
    Py_ssize_t window,
    Py_ssize_t exclude,
    double *mean_x,
    double *std_x,
    complex *x_buffer,
    complex *y_buffer,
    double *dist_buffer,
    double *mp,
    Py_ssize_t *mpi,
) nogil