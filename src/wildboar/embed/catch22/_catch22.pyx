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


cdef extern from "catch22.h":

    cdef double histogram_mode(
        double *x, size_t length, size_t *hist_counts, double *bin_edges, size_t n_bins
    ) nogil

    cdef double histogram_ami_even(
        double *x, size_t length, size_t tau, size_t n_bins
    ) nogil

    cdef double transition_matrix_ac_sumdiagcov(
            double *x, double *ac, size_t length, size_t n_groups
    ) nogil

    cdef double periodicity_wang_th0_01(double *x, size_t length) nogil

    cdef double embed2_dist_tau_d_expfit_meandiff(
        double *x, double *ac, size_t length
    ) nogil

    cdef double auto_mutual_info_stats_gaussian_fmmi(
        double *x, size_t length, size_t tau
    ) nogil

    cdef double outlier_include_np_mdrmd(
        double *x, size_t length, int sign, double inc
    ) nogil

    cdef double summaries_welch_rect(
        double *x, size_t length, int what, double *S, double *f, size_t n_welch
    ) nogil

cdef double transition_matrix_3ac_sumdiagcov(
    double *x,
    double *ac,
    Py_ssize_t length,
) nogil:
    return transition_matrix_ac_sumdiagcov(x, ac, length, 3)

#cdef double auto_mutual_info_stats_40_gaussian_fmmi(double *x, size_t length) nogil:
#    return auto_mutual_info_stats_gaussian_fmmi(x, length, 40)

cdef double _histogram_ami_even(
    double *x,
    Py_ssize_t length,
    size_t tau,
    size_t n_bins,
) nogil:
    return histogram_ami_even(x, length, tau, n_bins)

cdef double histogram_ami_even_2_5(double *x, Py_ssize_t length) nogil:
    return _histogram_ami_even(x, length, 2, 5)

cdef double _histogram_mode(
    double *x,
    Py_ssize_t length,
    size_t *bin_count,
    double *bin_edges,
    size_t n_bins,
) nogil:
    return histogram_mode(x, <size_t>length, bin_count, bin_edges, n_bins)

cdef double histogram_mode10(
        double *x,
        Py_ssize_t length,
        size_t *bin_count,
        double *bin_edges
) nogil:
    return _histogram_mode(x, length, bin_count, bin_edges, 10)

cdef double histogram_mode5(
        double *x,
        Py_ssize_t length,
        size_t *bin_count,
        double *bin_edges
) nogil:
    return _histogram_mode(x, length, bin_count, bin_edges, 5)

#
#
#
# # Inspired by: https://github.com/chlubba/catch22/blob/master/C/histcounts.c
# cdef void histcount(
#     Py_ssize_t stride,
#     double *x,
#     Py_ssize_t length,
#     Py_ssize_t *bin_count,
#     double *bin_edges,
#     Py_ssize_t n_bins,
# ) nogil:
#     cdef double min_val = INFINITY
#     cdef double max_val = -INFINITY
#     cdef Py_ssize_t i
#     cdef int bin
#     cdef double v, bin_width
#
#     for i in range(length):
#         v = x[i * stride]
#         if v < min_val:
#             min_val = v
#         if v > max_val:
#             max_val = v
#
#     memset(bin_count, 0, sizeof(Py_ssize_t) * n_bins)
#     if min_val == max_val:
#         bin_count[0] = length
#     else:
#         bin_width = (max_val - min_val) / n_bins
#         for i in range(length):
#             bin = <int>((x[i * stride] - min_val) / bin_width)
#             if bin < 0:
#                 bin = 0
#             if bin >= n_bins:
#                 bin = n_bins - 1
#             bin_count[bin] += 1
#
#     if bin_edges != NULL:
#         i = 0
#         for i in range(n_bins + 1):
#             bin_edges[i] = i * bin_width + min_val
#
#
# # Inspired by: https://github.com/chlubba/catch22/blob/master/C/DN_HistogramMode_5.c
# cdef double histogram_mode(
#     Py_ssize_t stride,
#     double *x,
#     Py_ssize_t length,
#     Py_ssize_t *bin_count,
#     double *bin_edges,
#     Py_ssize_t n_bins,
# ) nogil:
#     histcount(stride, x, length, bin_count, bin_edges, n_bins)
#
#     cdef Py_ssize_t i
#     cdef double max_bin = 0
#     cdef double num_max = 1
#     cdef double value = 0
#
#     for i in range(n_bins):
#         if bin_count[i] > max_bin:
#             max_bin = bin_count[i]
#             num_max = 1
#             value = (bin_edges[i] + bin_edges[i + 1]) / 2.0
#         elif bin_count[i] == max_bin:
#             num_max += 1
#             value += (bin_edges[i] + bin_edges[i + 1]) / 2.0
#
#     return value / num_max


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
    cdef double value = 0

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
            if stretch >= longest:
                longest = stretch
            stretch = 0
        else:
            stretch += 1

    if stretch > longest:
        return stretch
    else:
        return longest


cdef double local_mean_tauresrat(double *x, double *ac, Py_ssize_t n, Py_ssize_t lag) nogil:
    if n <= lag:
        return 0.0
    cdef:
        Py_ssize_t i
        Py_ssize_t j
        double lag_sum
        Py_ssize_t lag_out, out
        double *lag_ac

    lag_ac = <double*> malloc(sizeof(double) * n - lag)
    for i in range(n - lag):
        lag_sum = 0
        for j in range(lag):
            lag_sum += x[i + j]

        lag_ac[i] = x[i + lag] - lag_sum / lag

    lag_out = 0
    _stats.auto_correlation(lag_ac, n - lag, lag_ac)
    while lag_ac[lag_out] > 0 and lag_out < n - lag:
        lag_out += 1
    free(lag_ac)

    out = 0
    while ac[out] > 0 and out < n:
        out += 1

    return <double> lag_out / <double> out

























