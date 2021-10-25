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
from libc.math cimport INFINITY, ceil, exp, fabs, floor, pow, sqrt
from libc.stdlib cimport free, malloc, qsort
from libc.string cimport memcpy, memset

from wildboar.utils cimport _stats
from wildboar.utils._fft cimport _pocketfft


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
cdef double histogram_mode(
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


cdef double f1ecac(double *ac, Py_ssize_t n) nogil:
    cdef double threshold = 1.0 / exp(1.0)
    cdef Py_ssize_t i
    for i in range(n - 1):
        if (ac[i] - threshold) * (ac[i + 1] - threshold) < 0:
            return i + 1
    return n


cdef double first_min(double *ac, Py_ssize_t n) nogil:
    cdef Py_ssize_t i
    for i in range(1, n - 1):
        if ac[i] < ac[i - 1] and ac[i] < ac[i + 1]:
            return i
    return n

cdef double trev_1_num(Py_ssize_t stride, double *x, Py_ssize_t n) nogil:
    cdef Py_ssize_t i
    cdef double sum = 0
    for i in range(n - 1):
        sum += pow(x[i + 1] - x[i], 3.0)
    return sum / (n - 1)

cdef double local_mean_std(Py_ssize_t stride, double *x, Py_ssize_t n, Py_ssize_t lag) nogil:
    if n <= lag:
        return 0.0
    cdef _stats.IncStats inc_stats
    _stats.inc_stats_init(&inc_stats)
    cdef Py_ssize_t i, j
    cdef double lag_sum
    for i in range(n - lag):
        lag_sum = 0.0
        for j in range(lag):
            lag_sum += x[i * stride + j * stride]

        _stats.inc_stats_add(
            &inc_stats, 1.0, x[i * stride + lag * stride] - lag_sum / lag
        )

    return sqrt(_stats.inc_stats_variance(&inc_stats))

cdef double hrv_classic_pnn(Py_ssize_t stride, double *x, Py_ssize_t n, double pnn) nogil:
    cdef Py_ssize_t i
    cdef double value

    for i in range(1, n):
        if fabs(x[stride * i] - x[stride * (i - 1)]) * 1000 > pnn:
            value += 1

    return value / (n - 1)

cdef double above_mean_stretch(Py_ssize_t stride, double *x, Py_ssize_t n) nogil:
    cdef double mean = _stats.mean(stride, x, n)
    cdef double stretch = 0
    cdef double longest = 0
    cdef Py_ssize_t i

    for i in range(1, n):
        if x[i * stride] - mean <= 0 and x[(i - 1) * stride] - mean <= 0:
            if stretch > longest:
                longest = stretch
            stretch = 1
        else:
            stretch += 1

    if stretch > longest:
        return stretch
    else:
        return longest


cdef int vcmp(void *a , void *b) nogil:
    cdef double a_v = (<double*> a)[0]
    cdef double b_v = (<double*> b)[0]
    return <int>(a_v - b_v)


cdef double _find_quantile(double *x, Py_ssize_t n, double quant) nogil:
    cdef double limit = 0.5 / n
    cdef double index
    cdef Py_ssize_t left, right
    if quant < limit:
        return x[0]
    elif quant > limit:
        return x[n - 1]
    else:
        index = n * quant - 0.5
        left = <Py_ssize_t> floor(index)
        right = <Py_ssize_t> ceil(index)
        return x[left] + (left + right) * (x[right] - x[left]) / (right - left)


cdef void sb_coarse(double *x, Py_ssize_t n, Py_ssize_t ng, Py_ssize_t *labels) nogil:
    cdef double *tmp = <double*> malloc(sizeof(double) * n)
    memcpy(tmp, x, sizeof(double) * n)
    qsort(tmp, n, sizeof(double), &vcmp)
    cdef double step_size = 1 / (ng - 1);
    cdef double step_value = 0
    cdef double *quantile = <double*> malloc(sizeof(double) * ng + 1)
    cdef Py_ssize_t i, j
    for i in range(ng + 1):
        quantile[i] = _find_quantile(tmp, n, step_value)
        step_value += step_size

    quantile[0] -= 1
    for i in range(ng):
        for j in range(n):
            if quantile[i] < x[j] <= quantile[i + j]:
                labels[j] = i
    free(tmp)
    free(quantile)


cdef double transition_matrix_3ac_sumdiagcov(double *x, double *ac, Py_ssize_t n) nogil:
    cdef Py_ssize_t tau = 0

    # find the index of the first negative auto correlation
    while ac[tau] > 0 and tau < n:
        tau += 1

    cdef Py_ssize_t n_neg = (n - 1) // tau + 1
    cdef Py_ssize_t i, j
    cdef double *neg = <double*> malloc(sizeof(double) * n_neg)
    for i in range(n_neg):
        neg[i] = x[i * tau]

    cdef Py_ssize_t *labels = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * n_neg)
    sb_coarse(x, n, 3, labels)

    cdef double T[3][3]
    memset(&T, 0, sizeof(double) * 3 * 3)

    for j in range(n_neg - 1):
        T[labels[j]][labels[j + 1]] += 1




# def histogram_mode(np.ndarray x, n_bins):
#     cdef Py_ssize_t *bin_count
#     cdef double *bin_edges
#     cdef double result
#     bin_count = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * n_bins)
#     bin_edges = <double*> malloc(sizeof(double) * (n_bins + 1))
#     result = histogram_mode(
#         <Py_ssize_t> (x.strides[0] / <Py_ssize_t> x.itemsize),
#         <double*> x.data,
#         x.shape[0],
#         bin_count,
#         bin_edges,
#         n_bins
#     )
#     free(bin_count)
#     free(bin_edges)
#     return result



























