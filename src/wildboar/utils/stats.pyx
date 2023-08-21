# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np

from libc.math cimport INFINITY, floor, sqrt
from libc.stdlib cimport free, malloc
from libc.string cimport memset
from scipy.linalg.cython_blas cimport dnrm2

from ._fft cimport _pocketfft


cdef double EPSILON = 1e-13

cdef void fast_mean_std(
    double* data,
    Py_ssize_t length,
    double *mean,
    double* std,
) nogil:
    """Update the mean and standard deviation"""
    cdef double ex = 0
    cdef double ex2 = 0
    cdef Py_ssize_t i
    for i in range(length):
        current_value = data[i]
        ex += current_value
        ex2 += current_value ** 2

    mean[0] = ex / length
    ex2 = ex2 / length - mean[0] * mean[0]
    if ex2 > EPSILON:
        std[0] = sqrt(ex2)
    else:
        std[0] = 0.0


cdef void inc_stats_init(IncStats *self) nogil:
    self.mean = 0.0
    self.n_samples = 0.0
    self.sum_square = 0.0
    self.sum = 0.0

cdef void inc_stats_add(IncStats *self, double weight, double value) nogil:
    cdef double next_m
    self.n_samples += weight
    next_m = self.mean + (value - self.mean) / self.n_samples
    self.sum_square += (value - self.mean) * (value - next_m)
    self.mean = next_m
    self.sum += weight * value

cdef void inc_stats_remove(IncStats *self, double weight, double value) nogil:
    cdef double old_m
    if self.n_samples == 1.0:
        self.n_samples = 0.0
        self.mean = 0.0
        self.sum_square = 0.0
    else:
        old_m = (self.n_samples * self.mean - value) / (self.n_samples - weight)
        self.sum_square -= (value - self.mean) * (value - old_m)
        self.mean = old_m
        self.n_samples -= weight
    self.sum -= weight * value

cdef double inc_stats_n_samples(IncStats *self) nogil:
    return self.n_samples

cdef double inc_stats_sum(IncStats *self) nogil:
    return self.sum

cdef double inc_stats_mean(IncStats *self) nogil:
    return self.mean

cdef double inc_stats_variance(IncStats *self, bint sample=False) nogil:
    cdef double n_samples, var
    if sample:
        n_samples = self.n_samples - 1
    else:
        n_samples = self.n_samples
    if n_samples <= 1:
        return 0 
    else:
        var = self.sum_square / n_samples
        if var < EPSILON:
             var = 0.0
        return var

cdef void cumulative_mean_std(
    double *x,
    Py_ssize_t x_length, 
    Py_ssize_t y_length, 
    double *x_mean, 
    double *x_std
) nogil:
    cdef Py_ssize_t i
    cdef IncStats stats
    inc_stats_init(&stats)
    for i in range(x_length):
        inc_stats_add(&stats, 1.0, x[i])
        if i >= y_length - 1:
            x_mean[i - (y_length - 1)] = stats.mean
            std = sqrt(inc_stats_variance(&stats))
            x_std[i - (y_length - 1)] = std 
            inc_stats_remove(&stats, 1.0, x[i - (y_length - 1)])

cdef double mean(double *x, Py_ssize_t length) nogil:
    cdef double v = 0.0
    cdef Py_ssize_t i
    for i in range(length):
        v += x[i]
    return v / length

cdef double variance(double *x, Py_ssize_t length) nogil:
    if length == 1:
        return 0.0

    cdef double avg = mean(x, length)
    cdef double sum = 0
    cdef double v = 0.0
    cdef Py_ssize_t i
    for i in range(length):
        v = x[i] - avg
        sum += v * v
    return sum / length

cdef double slope(double *x, Py_ssize_t length) nogil:
    if length == 1:
        return 0.0
    cdef double y_mean = (length + 1) / 2.0
    cdef double x_mean = 0
    cdef double mean_diff = 0
    cdef double mean_y_sqr = 0
    cdef Py_ssize_t i, j

    for i in range(length):
        j = i + 1
        mean_diff += x[i] * j
        x_mean += x[i]
        mean_y_sqr += j * j
    mean_diff /= length
    mean_y_sqr /= length
    x_mean /= length
    return (mean_diff - y_mean * x_mean) / (mean_y_sqr - y_mean ** 2)

cdef double find_min(double *x, Py_ssize_t n, Py_ssize_t *min_index=NULL) nogil:
    cdef double min_val = INFINITY
    cdef Py_ssize_t i

    for i in range(n):
        if x[i] < min_val:
            if min_index != NULL:
                min_index[0] = i
            min_val = x[i]

    return min_val

cdef double covariance(double *x, double *y, Py_ssize_t length) nogil:
    cdef:
        double sum_x = 0.0
        double sum_y = 0.0
        double sum_xy = 0.0
        double k = 0.0
        double mean_x = 0.0
        double mean_y = 0.0
        double tmp_mean_x, tmp_mean_y, diff_x, diff_y
        Py_ssize_t i

    for i in range(length):
        tmp_mean_x = mean_x
        tmp_mean_y = mean_y
        diff_x = x[i] - tmp_mean_x
        diff_y = y[i] - tmp_mean_y
        k += 1
        sum_x += x[i]
        sum_y += y[i]
        sum_xy += diff_x * diff_y - diff_x * diff_y / k
        mean_x = sum_x / k
        mean_y = sum_y / k
    return sum_xy / k



cdef void _auto_correlation(double *x, Py_ssize_t n, double *out, complex *fft) nogil:
    cdef double avg = mean(x, n)
    cdef Py_ssize_t fft_length = n * 2 - 1
    cdef Py_ssize_t i
    for i in range(n):
        fft[i] = x[i] - avg
    for i in range(n, fft_length):
        fft[i] = 0.0

    _pocketfft.fft(fft, fft_length, 1.0)
    for i in range(fft_length):
        fft[i] = fft[i] * fft[i].conjugate()

    _pocketfft.ifft(fft, fft_length, 1.0)
    cdef complex first = fft[0]
    if first == 0j:
        first = 1j
    
    for i in range(n):
        out[i] = (fft[i] / first).real

cdef void auto_correlation(double *x, Py_ssize_t n, double *out) nogil:
    cdef Py_ssize_t fft_length = n * 2 - 1
    cdef complex *fft = <complex *> malloc(sizeof(complex) * fft_length)
    _auto_correlation(x, n, out, fft)
    free(fft)

cdef Py_ssize_t next_power_of_2(Py_ssize_t n) nogil:
    n = n - 1
    while n & n - 1:
        n = n & n - 1
    return n << 1

cdef int welch(
    double *x, 
    Py_ssize_t size, 
    int NFFT, 
    double Fs, 
    double *window, 
    int windowWidth,
    double *Pxx, 
    double *f,
) nogil:
    cdef double dt = 1.0 / Fs
    cdef double df = 1.0 / next_power_of_2(windowWidth) / dt
    cdef double m = mean(x, size)
    cdef int k = <int>floor(<double>size/(<double>windowWidth/2.0))-1;
    
    cdef int incx = 1
    cdef double w_norm = dnrm2(&windowWidth, window, &incx)
    cdef double KMU = k * w_norm * w_norm

    cdef double *P = <double*> malloc(sizeof(double) * NFFT)
    cdef complex *F = <complex*> malloc(sizeof(complex) * NFFT)
    cdef double pi
    cdef Py_ssize_t i, j, i_win

    for i in range(NFFT):
        P[i] = 0.0

    for i in range(k):
        i_win = <int>(i * <double>windowWidth/2.0)
        for j in range(windowWidth):
            F[j] = (window[j] * x[j + i_win]) - m

        for j in range(windowWidth, NFFT):
            F[j] = 0.0

        _pocketfft.fft(F, NFFT, 1.0)

        for j in range(NFFT):
            pi = abs(F[j])
            P[j] += pi * pi
    
    cdef Py_ssize_t n_out = NFFT // 2 + 1
    for i in range(n_out):
        Pxx[i] = P[i] / KMU * dt
        if i > 0 and i < n_out - 1:
            Pxx[i] *= 2
        f[i] = i * df

    free(P)
    free(F)
    return n_out
