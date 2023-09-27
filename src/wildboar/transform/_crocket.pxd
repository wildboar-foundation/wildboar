# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: initializedcheck=False

# Authors: Isak Samsten
# License: BSD 3 clause

cdef void apply_convolution(
    Py_ssize_t dilation,
    Py_ssize_t padding,
    double bias,
    double *weight,
    Py_ssize_t length,
    const double* x,
    Py_ssize_t x_length,
    double* mean_val,
    double* max_val,
) noexcept nogil
