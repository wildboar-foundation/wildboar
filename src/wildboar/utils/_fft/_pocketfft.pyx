# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

cdef extern from "pocketfft.h":
    ctypedef struct rfft_plan:
       pass

    ctypedef struct cfft_plan:
       pass

    cdef rfft_plan make_rfft_plan (size_t length) nogil
    cdef void destroy_rfft_plan (rfft_plan plan) nogil
    cdef int rfft_backward(rfft_plan plan, double *c, double fct) nogil
    cdef int rfft_forward(rfft_plan plan, double *c, double fct) nogil
    cdef size_t rfft_length(rfft_plan plan) nogil

    cdef cfft_plan make_cfft_plan(size_t length) nogil
    cdef void destroy_cfft_plan(cfft_plan plan) nogil
    cdef int cfft_backward(cfft_plan plan, double *c, double fct) nogil
    cdef int cfft_forward(cfft_plan plan, double *c, double fct) nogil
    cdef size_t cfft_length(cfft_plan plan) nogil

cdef void fft(complex *x, Py_ssize_t n, double fct) nogil:
    cdef cfft_plan fft_plan = make_cfft_plan(n)
    cfft_forward(fft_plan, <double *> x, fct)
    destroy_cfft_plan(fft_plan)

cdef void ifft(complex *x, Py_ssize_t n, double fct) nogil:
    cdef cfft_plan fft_plan = make_cfft_plan(n)
    cfft_backward(fft_plan, <double *> x, fct)
    destroy_cfft_plan(fft_plan)

cdef void rfft(double *x, Py_ssize_t n, double fct) nogil:
    cdef rfft_plan fft_plan = make_rfft_plan(n)
    rfft_forward(fft_plan, x, fct)
    destroy_rfft_plan(fft_plan)

cdef void irfft(double *x, Py_ssize_t n, double fct) nogil:
    cdef rfft_plan fft_plan = make_rfft_plan(n)
    rfft_backward(fft_plan, x, fct)
    destroy_rfft_plan(fft_plan)