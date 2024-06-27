# cython: language_level=3

from numpy cimport uint8_t, uint16_t, uint32_t

ctypedef fused sax_t:
    uint8_t
    uint16_t
    uint32_t

cdef double sax_distance(
    sax_t *x,
    sax_t *y,
    double *breakpoints,
    Py_ssize_t m,
    Py_ssize_t n,
) noexcept nogil
