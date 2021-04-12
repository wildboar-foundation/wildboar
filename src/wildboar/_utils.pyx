# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3

# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wildboar is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
# General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# Authors: Isak Samsten
from __future__ import print_function

import numpy as np

cimport numpy as np
from libc.math cimport M_PI, cos, log, log2, sin, sqrt
from libc.stdlib cimport realloc


cdef class RollingVariance:
    def __cinit__(self):
        self._m = 0.0
        self._n_samples = 0.0
        self._s = 0.0

    cdef void _reset(self) nogil:
        self._m = 0.0
        self._n_samples = 0.0
        self._s = 0.0
        self._sum = 0.0

    def reset(self):
        self._reset()

    cdef void _add(self, double weight, double value) nogil:
        cdef double next_m
        self._n_samples += weight
        next_m = self._m + (value - self._m) / self._n_samples
        self._s += (value - self._m) * (value - next_m)
        self._m = next_m
        self._sum += weight * value

    def add(self, value, weight=1.0):
        self._add(weight, value)

    cdef void _remove(self, double weight, double value) nogil:
        cdef double old_m
        if self._n_samples == 1.0:
            self._n_samples = 0.0
            self._m = 0.0
            self._s = 0.0
        else:
            old_m = (self._n_samples * self._m - value) / (self._n_samples - weight)
            self._s -= (value - self._m) * (value - old_m)
            self._m = old_m
            self._n_samples -= weight
        self._sum -= weight * value

    def remove(self, value, weight=1.0):
        self._remove(weight, value)

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def sum(self):
        return self._sum

    cdef double _mean(self) nogil:
        return self._m

    @property
    def mean(self):
        return self._mean()

    cdef double _variance(self) nogil:
        return 0.0 if self._n_samples <= 1 else self._s / self._n_samples

    @property
    def variance(self):
        return self._variance()

# For debugging
def print_tree(o, indent=1):
    if o.is_classification_leaf:
        print("-" * indent, "leaf: ")
        print("-" * indent, " proba: ", o.proba)
    elif o.is_regression_leaf:
        print("-" * indent, "leaf: ")
        print("-" * indent, " avg: ", o.regr_val)
    else:
        print("-" * indent, "branch:")
        print("-" * indent, " shapelet: ", o.shapelet.array)
        print("-" * indent, " threshold: ", o.threshold)
        print("-" * indent, " left:", end="\n")
        print_tree(o.left, indent + 1)
        print("-" * indent, " right:", end="\n")
        print_tree(o.right, indent + 1)

# For debugging
cdef void print_c_array_d(object name, double *arr, Py_ssize_t length):
    print(name, end=": ")
    for i in range(length):
        print(arr[i], end=" ")
    print()

# For debugging
cdef void print_c_array_i(object name, Py_ssize_t *arr, Py_ssize_t length):
    print(name, end=": ")
    for i in range(length):
        print(arr[i], end=" ")
    print()

cdef inline size_t rand_r(size_t *seed) nogil:
    """Returns a pesudo-random number based on the seed.

    :param seed: the initial seed (updated)
    :return: a psudo-random number
    """
    seed[0] = seed[0] * 1103515245 + 12345
    return seed[0] % (<size_t> RAND_R_MAX + 1)

cdef inline size_t rand_int(size_t min_val, size_t max_val, size_t *seed) nogil:
    """Returns a pseudo-random number in the range [`min_val` `max_val`[

    :param min_val: the minimum value
    :param max_val: the maximum value
    :param seed: the seed (updated)
    """
    if min_val == max_val:
        return min_val
    else:
        return min_val + rand_r(seed) % (max_val - min_val)

cdef inline double rand_uniform(double low, double high, size_t *random_state) nogil:
    """Generate a random double in the range [`low` `high`[."""
    return ((high - low) * <double> rand_r(random_state) / <double> RAND_R_MAX) + low

cdef inline double rand_normal(double mu, double sigma, size_t *random_state) nogil:
    cdef double x1, x2, w, _y1
    x1 = 2.0 * rand_uniform(0, 1, random_state) - 1.0
    x2 = 2.0 * rand_uniform(0, 1, random_state) - 1.0
    w = x1 * x1 + x2 * x2
    while w >= 1.0:
        x1 = 2.0 * rand_uniform(0, 1, random_state) - 1.0
        x2 = 2.0 * rand_uniform(0, 1, random_state) - 1.0
        w = x1 * x1 + x2 * x2

    w = sqrt( (-2.0 * log( w ) ) / w )
    _y1 = x1 * w
    y2 = x2 * w
    return mu + _y1 * sigma



# Implementation of introsort. Inspired by sklearn.tree
# implementation. This code is licensed under BSD3 (and not GPLv3)
#
# Including:
#  - argsort
#  - swap
#  - median3
#  - introsort
#  - sift_down
#  - heapsort


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


cdef void fast_mean_std(
        Py_ssize_t offset,
        Py_ssize_t stride,
        Py_ssize_t length,
        double* data,
        double *mean,
        double* std,
) nogil:
    """Update the mean and standard deviation"""
    cdef double ex = 0
    cdef double ex2 = 0
    cdef Py_ssize_t i
    for i in range(length):
        current_value = data[offset + i * stride]
        ex += current_value
        ex2 += current_value ** 2

    mean[0] = ex / length
    ex2 = ex2 / length - mean[0] * mean[0]
    if ex2 > 0:
        std[0] = sqrt(ex2)
    else:
        std[0] = 0.0


cpdef check_array_fast(np.ndarray x, bint ensure_2d=False, bint allow_nd=False, bint c_order=True):
    """Ensure that the array is valid and with dtype=np.float64.

    Parameters
    ----------
    x : ndarray
        The array to validate.

    ensure_2d : bool, optional
        Ensure that the array has 2 dimensions.

    allow_nd : bool, optional
        Allow more than 2 dimensions. Only valid if ensure_2d=False.

    c_order : bool, optional
        Ensure that the returned array is in row-major order.

    Returns
    -------
    x : ndarray
        Either a copy or the original array validated.
    """
    if ensure_2d:
        if x.ndim != 2:
            raise ValueError("not 2d, got %rd" % x.ndim)
    else:
        if not allow_nd and x.ndim > 1:
            raise ValueError("not 1d, got %rd" % x.ndim)

    if not x.flags.c_contiguous and c_order:
        x = np.ascontiguousarray(x, dtype=np.float64)
    if not x.dtype == np.float64:
        x = x.astype(np.float64)
    return x
