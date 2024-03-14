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
from ..utils._parallel import run_in_parallel
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


cdef class _FlatMatrixProfile:

    # safe write assigned i:s
    cdef double[:, :] mp
    cdef Py_ssize_t[:, :, :] mpi

    # already allocated
    cdef double[:, :] meanT
    cdef double[:, :] stdT

    # readonly
    cdef TSArray Q, T

    cdef Py_ssize_t window, dim, exclude

    def __cinit__(
        self,
        TSArray Q,
        TSArray T,
        double[:, :] meanT,
        double[:, :] stdT,
        Py_ssize_t window,
        Py_ssize_t dim,
        Py_ssize_t exclude,
        double[:, :] mp,
        Py_ssize_t[:, :, :] mpi,
    ):
        self.Q = Q
        self.T = T
        self.window = window
        self.dim = dim
        self.exclude = exclude
        self.mp = mp
        self.mpi = mpi
        self.meanT = meanT
        self.stdT = stdT

    @property
    def n_work(self):
        return self.Q.shape[0]

    def __call__(self, Py_ssize_t job_id, Py_ssize_t offset, Py_ssize_t batch_size):
        cdef Py_ssize_t i, j, k, l
        cdef IncStats stats
        cdef bint allow
        cdef Py_ssize_t profile_length = self.Q.shape[2] - self.window + 1

        # Only allocate window?
        cdef complex *Tbuf = <complex*> malloc(sizeof(complex) * self.T.shape[2])
        cdef complex *Qbuf = <complex*> malloc(sizeof(complex) * self.T.shape[2])

        # Only allocate T.shape[2] - window + 1?
        cdef double *dist_buffer = <double*> malloc(sizeof(double) * self.T.shape[2])

        with nogil:
            for i in range(offset, offset + batch_size):
                for j in range(profile_length):
                    self.mp[i, j] = INFINITY

                inc_stats_init(&stats)
                for j in range(self.window - 1):
                    inc_stats_add(&stats, 1.0, self.Q[i, self.dim, j])

                for j in range(profile_length):
                    inc_stats_add(&stats, 1.0, self.Q[i, self.dim, j + self.window - 1])
                    for k in range(self.T.shape[0]):
                        _mass_distance(
                            &self.T[k, self.dim, 0],
                            self.T.shape[2],
                            &self.Q[i, self.dim, j],
                            self.window,
                            stats.mean,
                            sqrt(inc_stats_variance(&stats)),
                            &self.meanT[k, 0],
                            &self.stdT[k, 0],
                            Tbuf,
                            Qbuf,
                            dist_buffer,
                        )

                        for l in range(self.T.shape[2] - self.window + 1):
                            allow = l <= j - self.exclude or l >= j + self.exclude

                            if dist_buffer[l] < self.mp[i, j] and allow:
                                self.mp[i, j] = dist_buffer[l]
                                self.mpi[i, j, 0] = k
                                self.mpi[i, j, 1] = l

                    inc_stats_remove(&stats, 1.0, self.Q[i, self.dim, j])
        free(Tbuf)
        free(Qbuf)
        free(dist_buffer)


def _flat_matrix_profile_join(
    TSArray Q,  # The needles
    TSArray T,  # The haystack
    Py_ssize_t window,
    Py_ssize_t dim,
    Py_ssize_t exclude,
    n_jobs,
):
    cdef Py_ssize_t profile_length = Q.shape[2] - window + 1
    cdef double[:, :] meanT = np.empty((T.shape[0], T.shape[2]), dtype=float)
    cdef double[:, :] stdT = np.empty((T.shape[0], T.shape[2]), dtype=float)
    cdef double[:,:] mp = np.empty((Q.shape[0] , profile_length), dtype=np.double)
    cdef Py_ssize_t[:,:,:] mpi = np.empty((Q.shape[0] , profile_length, 2), dtype=np.intp)

    cdef Py_ssize_t i
    for i in range(T.shape[0]):
        cumulative_mean_std(
            &T[i, dim, 0],
            T.shape[2],
            window,
            &meanT[i, 0],
            &stdT[i, 0],
        )

    run_in_parallel(
        _FlatMatrixProfile(
                Q,
                T,
                meanT,
                stdT,
                window,
                dim,
                exclude,
                mp,
                mpi,
        ),
        n_jobs=n_jobs,
        require="sharedmem"
    )

    return mp.base, mpi.base
