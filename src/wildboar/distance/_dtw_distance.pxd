# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

cdef struct DtwExtra:
    double *lower
    double *upper

cdef struct Deque:
    Py_ssize_t *queue
    int size
    int capacity
    int front
    int back

cdef void deque_init(Deque *c, Py_ssize_t capacity) nogil

cdef void deque_destroy(Deque *c) nogil

cdef void deque_push_back(Deque *c, Py_ssize_t v) nogil

cdef void deque_pop_front(Deque *c) nogil

cdef void deque_pop_back(Deque *c) nogil

cdef Py_ssize_t deque_front(Deque *c) nogil

cdef Py_ssize_t deque_back(Deque *c) nogil

cdef Py_ssize_t deque_size(Deque *c) nogil

cdef bint deque_empty(Deque *c) nogil

cdef void _dtw_align(
    double[:] A, 
    double[:] B, 
    Py_ssize_t warp_width, 
    double[:] weights, 
    double[:,:] out
) nogil

cdef double _dtw(
    double *X, 
    Py_ssize_t x_length, 
    double x_mean, 
    double x_std,
    double *Y, 
    Py_ssize_t y_length, 
    double y_mean, 
    double y_std,
    Py_ssize_t r, 
    double *cost, 
    double *cost_prev,
) nogil