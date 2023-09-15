# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

cdef struct IncStats:
    double n_samples
    double mean
    double sum_square
    double sum

cdef void inc_stats_init(IncStats *inc_stats) noexcept nogil

cdef void inc_stats_add(IncStats *inc_stats, double weight, double value) noexcept nogil

cdef void inc_stats_remove(IncStats *inc_stats, double weight, double value) noexcept nogil

cdef double inc_stats_mean(IncStats *inc_stats) noexcept nogil

cdef double inc_stats_variance(IncStats *inc_stats, bint sample=*) noexcept nogil

cdef double inc_stats_n_samples(IncStats *inc_stats) noexcept nogil

cdef double inc_stats_sum(IncStats *inc_stats) noexcept nogil

cdef void cumulative_mean_std(
    const double *x,
    Py_ssize_t x_length, 
    Py_ssize_t y_length, 
    double *x_mean, 
    double *x_std
) noexcept nogil

cdef double find_min(const double *x, Py_ssize_t n, Py_ssize_t *min_index=*) noexcept nogil

cdef double mean(const double *x, Py_ssize_t length) noexcept nogil

cdef double variance(const double *x, Py_ssize_t length) noexcept nogil

cdef double slope(const double *x, Py_ssize_t length) noexcept nogil

cdef double covariance(const double *x, const double *y, Py_ssize_t length) noexcept nogil

cdef void auto_correlation(const double *x, Py_ssize_t n, double *out) noexcept nogil

cdef void _auto_correlation(const double *x, Py_ssize_t n, double *out, complex *fft) noexcept nogil

cdef Py_ssize_t next_power_of_2(Py_ssize_t n) noexcept nogil

cdef int welch(
    const double *x, 
    Py_ssize_t size, 
    int NFFT, 
    double Fs, 
    double *window, 
    int windowWidth,
    double *Pxx, 
    double *f,
) noexcept nogil


cdef void fast_mean_std(
    const double* data,
    Py_ssize_t length,
    double *mean,
    double* std,
) noexcept nogil
