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

from wildboar.utils cimport stats


cdef extern from "catch22.h":

    cdef double histogram_mode(
        double *x, int length, int *hist_counts, double *bin_edges, int n_bins
    ) nogil

    cdef double histogram_ami_even(
        double *x, int length, int tau, int n_bins
    ) nogil

    cdef double transition_matrix_ac_sumdiagcov(
            double *x, double *ac, int length, int n_groups
    ) nogil

    cdef double periodicity_wang_th0_01(double *x, int length) nogil

    cdef double embed2_dist_tau_d_expfit_meandiff(
        double *x, double *ac, int length
    ) nogil

    cdef double auto_mutual_info_stats_gaussian_fmmi(
        double *x, int length, int tau
    ) nogil

    cdef double outlier_include_np_mdrmd(
        double *x, int length, int sign, double inc
    ) nogil

    cdef double summaries_welch_rect(
        double *x, int length, int what, double *S, double *f, int n_welch
    ) nogil

    cdef double motif_three_quantile_hh(double *x, int length) nogil

    cdef double fluct_anal_2_50_1_logi_prop_r1(
        double *y, int length, int lag, int how
    ) nogil


cdef double transition_matrix_3ac_sumdiagcov(
    double *x,
    double *ac,
    Py_ssize_t length,
) nogil:
    return transition_matrix_ac_sumdiagcov(x, ac, length, 3)

cdef double histogram_ami_even_2_5(double *x, Py_ssize_t length) nogil:
    return histogram_ami_even(x, length, 2, 5)


cdef double histogram_mode10(
        double *x,
        Py_ssize_t length,
        int *bin_count,
        double *bin_edges
) nogil:
    return histogram_mode(x, length, bin_count, bin_edges, 10)


cdef double histogram_mode5(
        double *x,
        Py_ssize_t length,
        int *bin_count,
        double *bin_edges
) nogil:
    return histogram_mode(x, length, bin_count, bin_edges, 5)


cdef double f1ecac(double *ac, Py_ssize_t n) nogil:
    if n <= 1:
        return 0.0

    cdef double threshold = 1.0 / exp(1.0)
    cdef Py_ssize_t i
    for i in range(n - 1):
        if (ac[i] - threshold) * (ac[i + 1] - threshold) < 0:
            return i + 1
    return n


cdef double first_min(double *ac, Py_ssize_t n) nogil:
    if n <= 2:
        return 0.0
    
    cdef Py_ssize_t i
    for i in range(1, n - 1):
        if ac[i] < ac[i - 1] and ac[i] < ac[i + 1]:
            return i
    return n


cdef double trev_1_num(double *x, Py_ssize_t n) nogil:
    if n <= 1:
        return 0.0

    cdef Py_ssize_t i
    cdef double sum = 0
    for i in range(n - 1):
        sum += pow(x[i + 1] - x[i], 3.0)
    return sum / (n - 1)


cdef double local_mean_std(double *x, Py_ssize_t n, Py_ssize_t lag) nogil:
    if n <= lag:
        return 0.0

    cdef stats.IncStats inc_stats
    stats.inc_stats_init(&inc_stats)
    cdef Py_ssize_t i, j
    cdef double lag_sum
    for i in range(n - lag):
        lag_sum = 0.0
        for j in range(lag):
            lag_sum += x[i + j]

        stats.inc_stats_add(
            &inc_stats, 1.0, x[i + lag] - lag_sum / lag
        )

    return sqrt(stats.inc_stats_variance(&inc_stats))


cdef double hrv_classic_pnn(double *x, Py_ssize_t n, double pnn) nogil:
    if n <= 1:
        return 0.0
    
    cdef Py_ssize_t i
    cdef double value = 0

    for i in range(1, n):
        if fabs(x[i] - x[(i - 1)]) * 1000 > pnn:
            value += 1

    return value / (n - 1)


cdef double above_mean_stretch(double *x, Py_ssize_t n) nogil:
    cdef double mean = stats.mean(x, n)
    cdef double stretch = 0
    cdef double longest = 0
    cdef Py_ssize_t i

    for i in range(n):
        if x[i] - mean <= 0:
            if stretch >= longest:
                longest = stretch
            stretch = 1
        else:
            stretch += 1

    if stretch > longest:
        return stretch
    else:
        return longest


cdef double below_diff_stretch(double *x, Py_ssize_t n) nogil:
    cdef double stretch = 0
    cdef double max_stretch = 0
    cdef double last_i = 0
    cdef Py_ssize_t i

    for i in range(n - 1):
        if x[i + 1] - x[i] >= 0 or i == n - 2:
            stretch = i - last_i
            if stretch > max_stretch:
                max_stretch = stretch
            last_i = i

    return max_stretch


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
    stats.auto_correlation(lag_ac, n - lag, lag_ac)
    while lag_ac[lag_out] > 0 and lag_out < n - lag:
        lag_out += 1
    free(lag_ac)

    out = 0
    while ac[out] > 0 and out < n:
        out += 1

    return <double> lag_out / <double> out

























