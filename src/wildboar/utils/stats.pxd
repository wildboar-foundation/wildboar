# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

cdef struct IncStats:
    double n_samples
    double mean
    double sum_square
    double sum

cdef void inc_stats_init(IncStats *inc_stats) nogil

cdef void inc_stats_add(IncStats *inc_stats, double weight, double value) nogil

cdef void inc_stats_remove(IncStats *inc_stats, double weight, double value) nogil

cdef double inc_stats_mean(IncStats *inc_stats) nogil

cdef double inc_stats_variance(IncStats *inc_stats, bint sample=*) nogil

cdef double inc_stats_n_samples(IncStats *inc_stats) nogil

cdef double inc_stats_sum(IncStats *inc_stats) nogil

cdef void cumulative_mean_std(
    double *x,
    Py_ssize_t x_length, 
    Py_ssize_t y_length, 
    double *x_mean, 
    double *x_std
) nogil

cdef double find_min(double *x, Py_ssize_t n, Py_ssize_t *min_index=*) nogil

cdef double mean(double *x, Py_ssize_t length) nogil

cdef double variance(double *x, Py_ssize_t length) nogil

cdef double slope(double *x, Py_ssize_t length) nogil

cdef double covariance(double *x, double *y, Py_ssize_t length) nogil

cdef void auto_correlation(double *x, Py_ssize_t n, double *out) nogil

cdef void _auto_correlation(double *x, Py_ssize_t n, double *out, complex *fft) nogil

cdef Py_ssize_t next_power_of_2(Py_ssize_t n) nogil

cdef int welch(
    double *x, 
    Py_ssize_t size, 
    int NFFT, 
    double Fs, 
    double *window, 
    int windowWidth,
    double *Pxx, 
    double *f,
) nogil


cdef void fast_mean_std(
    double* data,
    Py_ssize_t length,
    double *mean,
    double* std,
) nogil
