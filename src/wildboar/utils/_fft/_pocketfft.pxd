# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

cdef void fft(complex *x, Py_ssize_t n, double fct) noexcept nogil
cdef void ifft(complex *x, Py_ssize_t n, double fct) noexcept nogil

cdef void rfft(double *x, Py_ssize_t n, double fct) noexcept nogil
cdef void irfft(double *x, Py_ssize_t n, double fct) nogil