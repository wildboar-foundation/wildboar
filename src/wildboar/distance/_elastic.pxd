# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

cdef struct Deque:
    Py_ssize_t *queue
    int size
    int capacity
    int front
    int back

cdef void deque_init(Deque *c, Py_ssize_t capacity) nogil

cdef void deque_reset(Deque *c) nogil

cdef void deque_destroy(Deque *c) nogil

cdef void deque_push_back(Deque *c, Py_ssize_t v) nogil

cdef void deque_pop_front(Deque *c) nogil

cdef void deque_pop_back(Deque *c) nogil

cdef Py_ssize_t deque_front(Deque *c) nogil

cdef Py_ssize_t deque_back(Deque *c) nogil

cdef bint deque_empty(Deque *c) nogil

cdef Py_ssize_t deque_size(Deque *c) nogil
