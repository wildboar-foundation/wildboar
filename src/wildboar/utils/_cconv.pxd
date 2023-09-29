# cython: language_level=3

cdef void convolution_1d(
    Py_ssize_t stride,
    Py_ssize_t dilation,
    Py_ssize_t padding,
    double bias,
    double *kernel,
    Py_ssize_t k_len,
    const double* x,
    Py_ssize_t x_len,
    double* out,
) noexcept nogil
