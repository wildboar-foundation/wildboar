# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3

# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wildboar is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# Authors: Isak Samsten

from __future__ import print_function

from libc.math cimport log2
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
cdef void print_c_array_d(object name, double *arr, size_t length):
    print(name, end=": ")
    for i in range(length):
        print(arr[i], end=" ")
    print()

# For debugging
cdef void print_c_array_i(object name, size_t *arr, size_t length):
    print(name, end=": ")
    for i in range(length):
        print(arr[i], end=" ")
    print()

cdef size_t label_distribution(const size_t *samples,
                               const double *sample_weights,
                               size_t start,
                               size_t end,
                               const size_t *labels,
                               size_t label_stride,
                               size_t n_labels,
                               double *n_weighted_samples,
                               double *dist) nogil:
    """Computes the label distribution

    :param samples: the samples to include
    :param sample_weights: 
    :param start: the start position in samples
    :param end: the end position in samples
    :param labels: the labels
    :param label_stride: the stride in labels
    :param n_labels: the number labeles
    :param n_weighted_samples: (out) number of samples according to weight
    :param dist: (out) label distribution
    :return: number of classes included in the sample
    """
    cdef double sample_weight
    cdef size_t i, j, p, n_pos

    n_pos = 0
    n_weighted_samples[0] = 0
    for i in range(start, end):
        j = samples[i]
        p = j * label_stride

        if sample_weights != NULL:
            sample_weight = sample_weights[j]
        else:
            sample_weight = 1.0

        dist[labels[p]] += sample_weight
        n_weighted_samples[0] += sample_weight

    for i in range(n_labels):
        if dist[i] > 0:
            n_pos += 1

    return n_pos

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

cdef inline double rand_uniform(double low, double high, size_t*random_state) nogil:
    """Generate a random double in the range [`low` `high`[."""
    return ((high - low) * <double> rand_r(random_state) / <double> RAND_R_MAX) + low

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


cdef inline void argsort(double *values, size_t *samples, size_t n) nogil:
    if n == 0:
        return
    cdef size_t maxd = 2 * <size_t> log2(n)
    introsort(values, samples, n, maxd)

cdef inline void swap(double *values, size_t *samples,
                      size_t i, size_t j) nogil:
    values[i], values[j] = values[j], values[i]
    samples[i], samples[j] = samples[j], samples[i]

cdef inline double median3(double *values, size_t n) nogil:
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

cdef void introsort(double *values, size_t *samples,
                    size_t n, size_t maxd) nogil:
    cdef double pivot, value
    cdef size_t i, l, r

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

cdef inline void sift_down(double *values, size_t *samples,
                           size_t start, size_t end) nogil:
    cdef size_t child, maxind, root
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

cdef void heapsort(double *values, size_t *samples, size_t n) nogil:
    cdef size_t start, end

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

cdef int realloc_array(void** ptr, size_t old_size, size_t ptr_size, size_t *capacity)  nogil except -1:
    cdef void *tmp = ptr[0]
    if old_size >= capacity[0]:
        capacity[0] = old_size * 2
        tmp = realloc(ptr[0], ptr_size * capacity[0])
        if tmp == NULL:
            return -1
    ptr[0] = tmp
    return 0

cdef int safe_realloc(void** ptr, size_t new_size) nogil except -1:
    cdef void *tmp = ptr[0]
    tmp = realloc(ptr[0], new_size)
    if tmp == NULL:
        return -1

    ptr[0] = tmp
    return 0
