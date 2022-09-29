# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np

cimport numpy as np
from libc.math cimport M_PI, cos, log, log2, sin, sqrt
from libc.stdlib cimport realloc


cdef extern from "Python.h":
  cdef void* PyList_GET_ITEM(list, Py_ssize_t index) nogil

cdef class CList:
    
    def __cinit__(self, list py_list):
        self.py_list = py_list
        self.size = len(py_list)

    cdef void* get(self, Py_ssize_t i) nogil:
        return PyList_GET_ITEM(self.py_list, i)


cdef void strided_copy(Py_ssize_t stride, double *f, double *t, Py_ssize_t length) nogil:
    cdef Py_ssize_t i
    for i in range(length):
        t[i] = f[i * stride]


cdef inline void argsort(double *values, Py_ssize_t *samples, Py_ssize_t n) nogil:
    if n == 0:
        return
    cdef Py_ssize_t maxd = 2 * <Py_ssize_t> log2(n)
    introsort(values, samples, n, maxd)

cdef inline void swap(double *values, Py_ssize_t *samples,
                      Py_ssize_t i, Py_ssize_t j) nogil:
    values[i], values[j] = values[j], values[i]
    samples[i], samples[j] = samples[j], samples[i]

cdef inline double median3(double *values, Py_ssize_t n) nogil:
    cdef double a = values[0]
    cdef double b = values[n / 2]
    cdef double c = values[n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b

cdef void introsort(double *values, Py_ssize_t *samples,
                    Py_ssize_t n, Py_ssize_t maxd) nogil:
    cdef double pivot, value
    cdef Py_ssize_t i, l, r

    while n > 1:
        if maxd <= 0:
            heapsort(values, samples, n)
            return
        maxd -= 1

        pivot = median3(values, n)

        i = l = 0
        r = n
        while i < r:
            value = values[i]
            if value < pivot:
                swap(values, samples, i, l)
                i += 1
                l += 1
            elif value > pivot:
                r -= 1
                swap(values, samples, i, r)
            else:
                i += 1

        introsort(values, samples, l, maxd)
        values += r
        samples += r
        n -= r

cdef inline void sift_down(double *values, Py_ssize_t *samples,
                           Py_ssize_t start, Py_ssize_t end) nogil:
    cdef Py_ssize_t child, maxind, root
    root = start
    while True:
        child = root * 2 + 1
        maxind = root
        if child < end and values[maxind] < values[child]:
            maxind = child
        if child + 1 < end and values[maxind] < values[child + 1]:
            maxind = child + 1

        if maxind == root:
            break
        else:
            swap(values, samples, root, maxind)
            root = maxind

cdef void heapsort(double *values, Py_ssize_t *samples, Py_ssize_t n) nogil:
    cdef Py_ssize_t start, end

    start = (n - 2) / 2
    end = n
    while True:
        sift_down(values, samples, start, end)
        if start == 0:
            break
        start -= 1

    end = n - 1
    while end > 0:
        swap(values, samples, 0, end)
        sift_down(values, samples, 0, end)
        end = end - 1

cdef int realloc_array(void** ptr, Py_ssize_t old_size, Py_ssize_t ptr_size, Py_ssize_t *capacity)  nogil except -1:
    cdef void *tmp = ptr[0]
    if old_size >= capacity[0]:
        capacity[0] = old_size * 2
        tmp = realloc(ptr[0], ptr_size * capacity[0])
        if tmp == NULL:
            return -1
    ptr[0] = tmp
    return 0

cdef int safe_realloc(void** ptr, Py_ssize_t new_size) nogil except -1:
    cdef void *tmp = ptr[0]
    tmp = realloc(ptr[0], new_size)
    if tmp == NULL:
        return -1

    ptr[0] = tmp
    return 0

cdef np.ndarray to_ndarray_int(Py_ssize_t *arr, Py_ssize_t n):
    cdef Py_ssize_t i
    cdef np.ndarray out = np.zeros(n, dtype=int)
    for i in range(n):
        out[i] = arr[i]

    return out

cdef np.ndarray to_ndarray_double(double *arr, Py_ssize_t n):
    cdef Py_ssize_t i
    cdef np.ndarray out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        out[i] = arr[i]

    return out