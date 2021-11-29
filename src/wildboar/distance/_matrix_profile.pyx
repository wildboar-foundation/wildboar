cimport numpy as np

import numpy as np

from libc.math cimport INFINITY, sqrt
from libc.stdlib cimport free, malloc

from wildboar.utils.data cimport Dataset
from wildboar.utils.rand cimport RAND_R_MAX, shuffle
from wildboar.utils.stats cimport (
    IncStats,
    cumulative_mean_std,
    find_min,
    inc_stats_add,
    inc_stats_init,
    inc_stats_remove,
    inc_stats_variance,
)

from wildboar.utils.data import check_dataset

from ._mass cimport _mass_distance


cdef double EPSILON = 1e-13

cdef void _matrix_profile_stmp(
    double *x,
    Py_ssize_t x_length,
    double *y,
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
) nogil:
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


# cdef void _matrix_profile_stamp(
#     double *x,
#     Py_ssize_t x_length,
#     Py_ssize_t window,
#     Py_ssize_t exclude,
#     Py_ssize_t *order,
#     Py_ssize_t max_iter,
#     double *mean_x,
#     double *std_x,
#     complex *x_buffer,
#     complex *y_buffer,
#     double *dist_buffer,
#     double *mp,
#     Py_ssize_t *mpi,
# ) nogil:
#     """Compute the matrix profile using the STAMP algorithm

#     Parameters
#     ----------

#     x : double*
#         The time series.

#     x_length : int
#         The size of x.

#     y : double*
#         The time series.

#     y_length : int
#         The size of y.

#     window : int
#         The size of subsequences used to form the profile.

#     exclude : double
#         The size of the exclusion zone, expressed as a fraction of `window`.

#     order : int*
#         The order in which subsequences are evaluated, with size `profile_length`

#     max_iter : int 
#         The max iterations bounded by `profile_length`.

#     mean_x : double*
#         The buffer of cumulative mean of subsequences of size `window` in x, with size
#         `profile_length`.

#     std_x : double
#         The buffer of cumulative std of subsequences of size `window` in x, with size
#         `profile_length`.

#     x_buffer : complex*
#         The buffer used for the distance computation, with size `x_length`

#     y_buffer : complex*
#         The buffer used for the distance computation, with size `x_length`

#     dist_buffer : double*
#         The buffer used to store distances at the i:th iteration, with size
#         `profile_length`.

#     mp : double*
#         The matrix profile, with size `profile_length`

#     mpi : int*
#         The matrix profile index, with size `profile_length`

#     Notes
#     =====

#     - The `profile_length` is computed as `x_length - window + 1`
#     - The buffer-parameters can be empty on entry.
#     """
#     cdef Py_ssize_t i, j, k
#     cdef double std
#     cdef IncStats stats
#     cdef Py_ssize_t profile_length = y_length - window + 1
#     cumulative_mean_std(x, x_length, window, mean_x, std_x)
    
#     for i in range(profile_length):
#         mp[i] = INFINITY
#         mpi[i] = -1
    
#     for k in range(profile_length):
#         if k >= max_iter:
#             break
        
#         i = order[k]
#         inc_stats_init(&stats)
#         for j in range(window):
#             inc_stats_add(&stats, 1.0, y[i + j])

#         std = sqrt(inc_stats_variance(&stats))
#         _mass_distance(
#             x, 
#             x_length,
#             y + i,
#             window,
#             stats.mean,
#             std,
#             mean_x,
#             std_x,
#             x_buffer,
#             y_buffer,
#             dist_buffer,
#         )
#         for j in range(profile_length):
#             if dist_buffer[j] < mp[j]:# and (j <= i - exclude or j >= i + exclude):
#                 mp[j] = dist_buffer[j]
#                 mpi[j] = i


def _paired_matrix_profile(
    np.ndarray x,
    np.ndarray y,
    Py_ssize_t w,
    Py_ssize_t dim, 
    Py_ssize_t exclude, 
    n_jobs,
):
    x = check_dataset(x, allow_1d=True)
    y = check_dataset(y, allow_1d=True)
    cdef Dataset x_dataset = Dataset(x)
    cdef Dataset y_dataset = Dataset(y)
    cdef Py_ssize_t profile_length = y_dataset.n_timestep - w + 1
    cdef Py_ssize_t i
    cdef double *mean_x = <double*> malloc(sizeof(double) * x_dataset.n_timestep)
    cdef double *std_x = <double*> malloc(sizeof(double) * x_dataset.n_timestep)
    cdef double *dist_buffer = <double*> malloc(sizeof(double) * x_dataset.n_timestep)
    cdef complex *x_buffer = <complex*> malloc(sizeof(complex) * x_dataset.n_timestep)
    cdef complex *y_buffer = <complex*> malloc(sizeof(complex) * x_dataset.n_timestep)
    
    cdef np.ndarray mp = np.empty((x_dataset.n_samples, profile_length), dtype=np.double)
    cdef np.ndarray mpi = np.empty((x_dataset.n_samples, profile_length), dtype=np.intp)
    
    with nogil:
        for i in range(x_dataset.n_samples):
            _matrix_profile_stmp(
                x_dataset.get_sample(i, dim=dim),
                x_dataset.n_timestep,
                y_dataset.get_sample(i, dim=dim),
                y_dataset.n_timestep,
                w,
                exclude,
                mean_x, 
                std_x,
                x_buffer,
                y_buffer,
                dist_buffer,
                (<double*> mp.data) + i * profile_length,
                (<Py_ssize_t*> mpi.data) + i * profile_length,
            )
    free(mean_x)
    free(std_x)
    free(dist_buffer)
    free(x_buffer)
    free(y_buffer)
    return mp, mpi


def test(np.ndarray x):
    cdef np.ndarray d = np.zeros(x.size - 10 + 1, dtype=np.double)
    cdef np.ndarray i = np.zeros(x.size - 10 + 1, dtype=np.intp)
    n = x.size
    m = 10
    cdef double *mean_x = <double*> malloc(sizeof(double) * n)
    cdef double *std_x = <double*> malloc(sizeof(double) * n)
    cdef double *dist_buffer = <double*> malloc(sizeof(double) * n)
    cdef complex *x_buffer = <complex*> malloc(sizeof(complex) * n)
    cdef complex *y_buffer = <complex*> malloc(sizeof(complex) * n)
    _matrix_profile_stmp(
        <double*> x.data,
        n,
        <double*> x.data,
        n,
        m,
        3,
        mean_x,
        std_x,
        x_buffer,
        y_buffer,
        dist_buffer,
        <double*> d.data,
        <Py_ssize_t*> i.data,
    )
    return i, d
    