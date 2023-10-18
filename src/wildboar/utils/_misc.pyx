# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: initializedcheck=False

# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np

from libc.math cimport M_PI, cos, log, log2, sin, sqrt, INFINITY
from libc.stdlib cimport realloc, malloc, free
from libc.string cimport memset

from ..utils cimport TSArray

cdef void _heap_shift_down(
    HeapElement *heap, Py_ssize_t startpos, Py_ssize_t pos
) noexcept nogil:
    cdef HeapElement newelement = heap[pos]
    cdef Py_ssize_t parentpos
    while pos > startpos:
        parentpos = (pos - 1) >> 1  # // 2
        if newelement.value > heap[parentpos].value:
            heap[pos] = heap[parentpos]
            pos = parentpos
            continue
        break

    heap[pos] = newelement

cdef void _heap_shift_up(
    HeapElement *heap, Py_ssize_t pos, Py_ssize_t endpos
) noexcept nogil:
    cdef Py_ssize_t startpos = pos
    cdef HeapElement newelement = heap[pos]
    cdef Py_ssize_t childpos = 2 * pos + 1
    cdef Py_ssize_t rightpos
    while childpos < endpos:
        rightpos = childpos + 1
        if rightpos < endpos and heap[childpos].value < heap[rightpos].value:
            childpos = rightpos

        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2 * pos + 1

    heap[pos] = newelement
    _heap_shift_down(heap, startpos, pos)

# A Heap implementation that retains the `k` smallest elements.
# The first element of the heap is the largest of the `k` smallest
# elements and can be retreived in O(1) time. We maintain a separate
# datastructure for the minimum value so that can also be retreived
# in constant time.
#
# `_min` returns the smallest value among k smallest elements.
# `_max` returns the largest value among the top k elements.
#
# The data structure efficently act as a proxy for `argpartition` in numpy.
cdef class Heap:

    def __cinit__(self, Py_ssize_t max_elements):
        self.heap = <HeapElement*> malloc(sizeof(HeapElement) * max_elements)
        self.n_elements = 0
        self.max_elements = max_elements

    def __dealloc__(self):
        if self.heap != NULL:
            free(self.heap)
            self.heap = NULL

    cdef void push(self, Py_ssize_t index, double value) noexcept nogil:
        cdef HeapElement element
        element.index = index
        element.value = value

        if self.n_elements == 0:
            self.heap[0] = element
            self.n_elements += 1
        else:
            if self.n_elements < self.max_elements:
                self.heap[self.n_elements] = element
                self.n_elements += 1
                _heap_shift_down(self.heap, 0, self.n_elements - 1)
            elif self.heap[0].value > element.value:
                self.heap[0] = element
                _heap_shift_up(self.heap, 0, self.max_elements)

    cdef HeapElement maxelement(self) noexcept nogil:
        return self.heap[0]

    cdef double maxvalue(self) noexcept nogil:
        return self.heap[0].value

    cdef HeapElement getelement(self, Py_ssize_t i) noexcept nogil:
        return self.heap[i]

    cdef void reset(self) noexcept nogil:
        self.n_elements = 0

    cdef bint isempty(self) noexcept nogil:
        return self.n_elements == 0

    cdef bint isfull(self) noexcept nogil:
        return self.n_elements == self.max_elements


cdef extern from "Python.h":
    cdef void* PyList_GET_ITEM(list, Py_ssize_t index) nogil

cdef class List:

    def __init__(self, list py_list):
        self.py_list = py_list
        self.size = len(py_list)

    cdef void* get(self, Py_ssize_t i) noexcept nogil:
        return PyList_GET_ITEM(self.py_list, i)


cdef inline void argsort(
    double_or_int *values, Py_ssize_t *samples, Py_ssize_t n
) noexcept nogil:
    if n == 0:
        return
    cdef Py_ssize_t maxd = 2 * <Py_ssize_t> log2(n)
    introsort(values, samples, n, maxd)

cdef inline void swap(
    double_or_int *values,
    Py_ssize_t *samples,
    Py_ssize_t i,
    Py_ssize_t j
) noexcept nogil:
    values[i], values[j] = values[j], values[i]
    samples[i], samples[j] = samples[j], samples[i]

cdef inline double_or_int median3(double_or_int *values, Py_ssize_t n) noexcept nogil:
    cdef double_or_int a = values[0]
    cdef double_or_int b = values[n / 2]
    cdef double_or_int c = values[n - 1]
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

cdef void introsort(
    double_or_int *values,
    Py_ssize_t *samples,
    Py_ssize_t n,
    Py_ssize_t maxd
) noexcept nogil:
    cdef double_or_int pivot, value
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

cdef inline void sift_down(double_or_int *values, Py_ssize_t *samples,
                           Py_ssize_t start, Py_ssize_t end) noexcept nogil:
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

cdef void heapsort(
    double_or_int *values, Py_ssize_t *samples, Py_ssize_t n
) noexcept nogil:
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

cdef int realloc_array(
    void** ptr, Py_ssize_t old_size, Py_ssize_t ptr_size, Py_ssize_t *capacity
)  except -1 nogil:
    cdef void *tmp = ptr[0]
    if old_size >= capacity[0]:
        capacity[0] = old_size * 2
        tmp = realloc(ptr[0], ptr_size * capacity[0])
        if tmp == NULL:
            return -1
    ptr[0] = tmp
    return 0

cdef int safe_realloc(void** ptr, Py_ssize_t new_size) except -1 nogil:
    cdef void *tmp = ptr[0]
    tmp = realloc(ptr[0], new_size)
    if tmp == NULL:
        return -1

    ptr[0] = tmp
    return 0

cdef object to_ndarray_int(Py_ssize_t *arr, Py_ssize_t n):
    cdef Py_ssize_t i
    cdef Py_ssize_t[:] out = np.zeros(n, dtype=np.intp)
    for i in range(n):
        out[i] = arr[i]

    return out.base

cdef object to_ndarray_double(double *arr, Py_ssize_t n):
    cdef Py_ssize_t i
    cdef double[:] out = np.zeros(n, dtype=float)
    for i in range(n):
        out[i] = arr[i]

    return out.base


def _test_ts_array(TSArray arr):
    return arr[0, 0, 0]
