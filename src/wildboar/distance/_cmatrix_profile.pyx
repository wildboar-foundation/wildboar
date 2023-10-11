# cython: language_level=3
# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: initializedcheck=False

# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np

from libc.math cimport INFINITY, sqrt
from libc.stdlib cimport free, malloc

from ..utils cimport TSArray
from ..utils._rand cimport RAND_R_MAX
from ..utils._stats cimport (
    IncStats,
    cumulative_mean_std,
    find_min,
    inc_stats_add,
    inc_stats_init,
    inc_stats_remove,
    inc_stats_variance,
)
from ._mass cimport _mass_distance


cdef double EPSILON = 1e-13

cdef void _matrix_profile_stmp(
    const double *x,
    Py_ssize_t x_length,
    const double *y,
    Py_ssize_t y_length,
    Py_ssize_t window,
    Py_ssize_t exclude,
    double *mean_x,
    double *std_x,
    complex *x_buffer,
    complex *y_buffer,
    double *dist_buffer,
    double *mp,
    Py_ssize_t *mpi,
) noexcept nogil:
    """Compute the matrix profile using the STMP algorithm

    Parameters
    ----------
    x : double*
        The time series.
    x_length : int
        The size of x.
    y : double*
        A time series.
    y_length : int
        The size of y
    window : int
        The size of subsequences used to form the profile.
    exclude : double
        The size of the exclusion zone, expressed as a fraction of `window`.
    mean_x : double*
        The buffer of cumulative mean of subsequences of size `window` in x, with size
        `profile_length`.
    std_x : double
        The buffer of cumulative std of subsequences of size `window` in x, with size
        `profile_length`.
    x_buffer : complex*
        The buffer used for the distance computation, with size `x_length`
    y_buffer : complex*
        The buffer used for the distance computation, with size `x_length`
    dist_buffer : double*
        The buffer used to store distances at the i:th iteration, with size
        `profile_length`.
    mp : double*
        The matrix profile, with size `profile_length`
    mpi : int*
        The matrix profile index, with size `profile_length`

    Notes
    =====
    - The `profile_length` is computed as `y_length - window + 1`
    - The buffer-parameters can be empty on entry.
    """
    cdef Py_ssize_t i, j
    cdef double std
    cdef IncStats stats
    cdef Py_ssize_t profile_length = y_length - window + 1
    cumulative_mean_std(x, x_length, window, mean_x, std_x)
    inc_stats_init(&stats)
    for i in range(window - 1):
        inc_stats_add(&stats, 1.0, y[i])

    for i in range(profile_length):
        mp[i] = INFINITY
        mpi[i] = -1

    for i in range(profile_length):
        inc_stats_add(&stats, 1.0, y[i + window - 1])
        std = sqrt(inc_stats_variance(&stats))
        _mass_distance(
            x,
            x_length,
            y + i,
            window,
            stats.mean,
            std,
            mean_x,
            std_x,
            x_buffer,
            y_buffer,
            dist_buffer,
        )
        inc_stats_remove(&stats, 1.0, y[i])
        for j in range(x_length - window + 1):
            if dist_buffer[j] < mp[i] and (j <= i - exclude or j >= i + exclude):
                mp[i] = dist_buffer[j]
                mpi[i] = j


def _paired_matrix_profile(
    TSArray X,
    TSArray Y,
    Py_ssize_t w,
    Py_ssize_t dim,
    Py_ssize_t exclude,
    n_jobs,
):
    cdef Py_ssize_t profile_length = Y.shape[2] - w + 1
    cdef Py_ssize_t i
    cdef double *mean_x = <double*> malloc(sizeof(double) * X.shape[2])
    cdef double *std_x = <double*> malloc(sizeof(double) * X.shape[2])
    cdef double *dist_buffer = <double*> malloc(sizeof(double) * X.shape[2])
    cdef complex *x_buffer = <complex*> malloc(sizeof(complex) * X.shape[2])
    cdef complex *y_buffer = <complex*> malloc(sizeof(complex) * X.shape[2])

    cdef double[:, :] mp = np.empty((X.shape[0], profile_length), dtype=np.double)
    cdef Py_ssize_t[:, :] mpi = np.empty((X.shape[0], profile_length), dtype=np.intp)

    with nogil:
        for i in range(X.shape[0]):
            _matrix_profile_stmp(
                &X[i, dim, 0],
                X.shape[2],
                &Y[i, dim, 0],
                Y.shape[2],
                w,
                exclude,
                mean_x,
                std_x,
                x_buffer,
                y_buffer,
                dist_buffer,
                &mp[i, 0],
                &mpi[i, 0],
            )
    free(mean_x)
    free(std_x)
    free(dist_buffer)
    free(x_buffer)
    free(y_buffer)
    return mp.base, mpi.base
