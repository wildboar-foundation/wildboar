# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: initializedcheck=False

# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np

from libc.math cimport INFINITY, ceil, exp, fabs, floor, pow, sqrt
from libc.stdlib cimport free, malloc, qsort
from libc.string cimport memcpy, memset

from ...utils cimport TSArray, _stats


cdef extern from "catch22.h":

    cdef double histogram_mode(
        const double *x, int length, int *hist_counts, double *bin_edges, int n_bins
    ) nogil

    cdef double histogram_ami_even(
        const double *x, int length, int tau, int n_bins
    ) nogil

    cdef double transition_matrix_ac_sumdiagcov(
        const double *x, double *ac, int length, int n_groups
    ) nogil

    cdef double periodicity_wang_th0_01(const double *x, int length) nogil

    cdef double embed2_dist_tau_d_expfit_meandiff(
        const double *x, double *ac, int length
    ) nogil

    cdef double auto_mutual_info_stats_gaussian_fmmi(
        const double *x, int length, int tau
    ) nogil

    cdef double outlier_include_np_mdrmd(
        const double *x, int length, int sign, double inc
    ) nogil

    cdef double summaries_welch_rect(
        const double *x, int length, int what, double *S, double *f, int n_welch
    ) nogil

    cdef double motif_three_quantile_hh(const double *x, int length) nogil

    cdef double fluct_anal_2_50_1_logi_prop_r1(
        const double *y, int length, int lag, int how
    ) nogil


cdef double transition_matrix_3ac_sumdiagcov(
    const double *x, double *ac, Py_ssize_t length,
) noexcept nogil:
    return transition_matrix_ac_sumdiagcov(x, ac, length, 3)


cdef double histogram_ami_even_2_5(const double *x, Py_ssize_t length) noexcept nogil:
    return histogram_ami_even(x, length, 2, 5)


cdef double histogram_mode10(
    const double *x, Py_ssize_t length, int *bin_count, double *bin_edges
) noexcept nogil:
    return histogram_mode(x, length, bin_count, bin_edges, 10)


cdef double histogram_mode5(
    const double *x, Py_ssize_t length, int *bin_count, double *bin_edges
) noexcept nogil:
    return histogram_mode(x, length, bin_count, bin_edges, 5)


cdef double f1ecac(const double *ac, Py_ssize_t n) noexcept nogil:
    if n <= 1:
        return 0.0

    cdef double threshold = 1.0 / exp(1.0)
    cdef Py_ssize_t i
    for i in range(n - 1):
        if (ac[i] - threshold) * (ac[i + 1] - threshold) < 0:
            return i + 1
    return n


cdef double first_min(double *ac, Py_ssize_t n) noexcept nogil:
    if n <= 2:
        return 0.0
    
    cdef Py_ssize_t i
    for i in range(1, n - 1):
        if ac[i] < ac[i - 1] and ac[i] < ac[i + 1]:
            return i
    return n


cdef double trev_1_num(const double *x, Py_ssize_t n) noexcept nogil:
    if n <= 1:
        return 0.0

    cdef Py_ssize_t i
    cdef double sum = 0
    for i in range(n - 1):
        sum += pow(x[i + 1] - x[i], 3.0)
    return sum / (n - 1)


cdef double local_mean_std(const double *x, Py_ssize_t n, Py_ssize_t lag) noexcept nogil:
    if n <= lag:
        return 0.0

    cdef _stats.IncStats inc_stats
    _stats.inc_stats_init(&inc_stats)
    cdef Py_ssize_t i, j
    cdef double lag_sum
    for i in range(n - lag):
        lag_sum = 0.0
        for j in range(lag):
            lag_sum += x[i + j]

        _stats.inc_stats_add(
            &inc_stats, 1.0, x[i + lag] - lag_sum / lag
        )

    return sqrt(_stats.inc_stats_variance(&inc_stats, True))


cdef double hrv_classic_pnn(const double *x, Py_ssize_t n, double pnn) noexcept nogil:
    if n <= 1:
        return 0.0
    
    cdef Py_ssize_t i
    cdef double value = 0

    for i in range(1, n):
        if fabs(x[i] - x[(i - 1)]) * 1000 > pnn:
            value += 1

    return value / (n - 1)


cdef double above_mean_stretch(const double *x, Py_ssize_t n) noexcept nogil:
    cdef double mean = _stats.mean(x, n)
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


cdef double below_diff_stretch(const double *x, Py_ssize_t n) noexcept nogil:
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


cdef double local_mean_tauresrat(
    const double *x, double *ac, Py_ssize_t n, Py_ssize_t lag
) noexcept nogil:
    if n <= lag or lag == 0:
        return 0.0
    cdef:
        Py_ssize_t i
        Py_ssize_t j
        double lag_sum
        Py_ssize_t lag_out, x_out
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

    x_out = 0
    while ac[x_out] > 0 and x_out < n:
        x_out += 1

    if x_out == 0:
        x_out = 1
    return <double> lag_out / <double> x_out
