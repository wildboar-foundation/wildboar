# cython: language_level=3

from numpy cimport uint8_t, uint16_t, uint32_t

ctypedef fused sax_t:
    uint8_t
    uint16_t
    uint32_t

cdef double sax_distance(
    const sax_t *x,
    const sax_t *y,
    const double *breakpoints,
    Py_ssize_t m,
    Py_ssize_t n,
) noexcept nogil
