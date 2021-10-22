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
#
# Authors: Isak Samsten

cimport numpy as np
from libc.math cimport INFINITY
from libc.stdlib cimport free, malloc
from libc.string cimport memset


# Inspired by: https://github.com/chlubba/catch22/blob/master/C/histcounts.c
cdef void histcount(
    Py_ssize_t stride,
    double *x,
    Py_ssize_t length,
    Py_ssize_t *bin_count,
    double *bin_edges,
    Py_ssize_t n_bins,
) nogil:
    cdef double min_val = INFINITY
    cdef double max_val = -INFINITY
    cdef Py_ssize_t i
    cdef int bin
    cdef double v, bin_width

    for i in range(length):
        v = x[i * stride]
        if v < min_val:
            min_val = v
        if v > max_val:
            max_val = v

    memset(bin_count, 0, sizeof(Py_ssize_t) * n_bins)
    if min_val == max_val:
        bin_count[0] = length
    else:
        bin_width = (max_val - min_val) / n_bins
        for i in range(length):
            bin = <int>((x[i * stride] - min_val) / bin_width)
            if bin < 0:
                bin = 0
            if bin >= n_bins:
                bin = n_bins - 1
            bin_count[bin] += 1

    if bin_edges != NULL:
        i = 0
        for i in range(n_bins + 1):
            bin_edges[i] = i * bin_width + min_val


# Inspired by: https://github.com/chlubba/catch22/blob/master/C/DN_HistogramMode_5.c
cdef double _histogram_mode(
    Py_ssize_t stride,
    double *x,
    Py_ssize_t length,
    Py_ssize_t *bin_count,
    double *bin_edges,
    Py_ssize_t n_bins,
) nogil:
    histcount(stride, x, length, bin_count, bin_edges, n_bins)

    cdef Py_ssize_t i
    cdef double max_bin = 0
    cdef double num_max = 1
    cdef double value = 0

    for i in range(n_bins):
        if bin_count[i] > max_bin:
            max_bin = bin_count[i]
            num_max = 1
            value = (bin_edges[i] + bin_edges[i + 1]) / 2.0
        elif bin_count[i] == max_bin:
            num_max += 1
            value += (bin_edges[i] + bin_edges[i + 1]) / 2.0

    return value / num_max


def histogram_mode(np.ndarray x, n_bins):
    cdef Py_ssize_t *bin_count
    cdef double *bin_edges
    cdef double result
    bin_count = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * n_bins)
    bin_edges = <double*> malloc(sizeof(double) * (n_bins + 1))
    result = _histogram_mode(
        <Py_ssize_t> (x.strides[0] / <Py_ssize_t> x.itemsize),
        <double*> x.data,
        x.shape[0],
        bin_count,
        bin_edges,
        n_bins
    )
    free(bin_count)
    free(bin_edges)
    return result



























