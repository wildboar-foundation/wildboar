# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

cimport numpy as np

import numpy as np

from libc.math cimport INFINITY, NAN, sqrt
from libc.stdlib cimport free, malloc

from ..utils._fft cimport _pocketfft
from ..utils.data cimport Dataset
from ..utils.stats cimport (
    IncStats,
    cumulative_mean_std,
    find_min,
    inc_stats_add,
    inc_stats_init,
    inc_stats_remove,
    inc_stats_variance,
)
from ._distance cimport (
    DistanceMeasure,
    ScaledSubsequenceDistanceMeasure,
    Subsequence,
    SubsequenceDistanceMeasure,
    SubsequenceView,
)


cdef double EPSILON = 1e-10

cdef class ScaledMassSubsequenceDistanceMeasure(ScaledSubsequenceDistanceMeasure):
    cdef double *mean_x
    cdef double *std_x
    cdef double *dist_buffer
    cdef complex *x_buffer
    cdef complex *y_buffer

    def __cinit__(self):
        self.mean_x = NULL
        self.std_x = NULL
        self.x_buffer = NULL
        self.y_buffer = NULL
    
    def __dealloc__(self):
        self.__free()

    def __reduce__(self):
        return self.__class__, ()
    
    cdef void __free(self) nogil:
        if self.mean_x != NULL:
            free(self.mean_x)
            self.mean_x = NULL
        if self.std_x != NULL:
            free(self.std_x)
            self.std_x = NULL
        if self.dist_buffer != NULL:
            free(self.dist_buffer)
            self.dist_buffer = NULL
        if self.x_buffer != NULL:
            free(self.x_buffer)
            self.x_buffer = NULL
        if self.y_buffer != NULL:
            free(self.y_buffer)
            self.y_buffer = NULL

    cdef int reset(self, Dataset dataset) nogil:
        self.__free() 
        self.x_buffer = <complex*> malloc(sizeof(complex) * dataset.n_timestep)
        self.y_buffer = <complex*> malloc(sizeof(complex) * dataset.n_timestep)
        self.mean_x = <double*> malloc(sizeof(double) * dataset.n_timestep)
        self.std_x = <double*> malloc(sizeof(double) * dataset.n_timestep)
        self.dist_buffer = <double*> malloc(sizeof(double) * dataset.n_timestep)
        return 0

    cdef double transient_distance(
        self,
        SubsequenceView *s,
        Dataset dataset,
        Py_ssize_t index,
        Py_ssize_t *return_index=NULL,
    ) nogil:
        cumulative_mean_std(
            dataset.get_sample(index, s.dim),
            dataset.n_timestep,
            s.length,
            self.mean_x,
            self.std_x,
        )
        _mass_distance(
            dataset.get_sample(index, s.dim),
            dataset.n_timestep,
            dataset.get_sample(s.index, s.dim) + s.start,
            s.length,
            s.mean,
            s.std,
            self.mean_x,
            self.std_x,
            self.x_buffer,
            self.y_buffer,
            self.dist_buffer,
        )
        return find_min(
            self.dist_buffer, dataset.n_timestep - s.length + 1, return_index
        )

    cdef double persistent_distance(
        self,
        Subsequence *s,
        Dataset dataset,
        Py_ssize_t index,
        Py_ssize_t *return_index=NULL,
    ) nogil:
        cumulative_mean_std(
            dataset.get_sample(index, s.dim),
            dataset.n_timestep,
            s.length,
            self.mean_x,
            self.std_x,
        )
        _mass_distance(
            dataset.get_sample(index, s.dim),
            dataset.n_timestep,
            s.data,
            s.length,
            s.mean,
            s.std,
            self.mean_x,
            self.std_x,
            self.x_buffer,
            self.y_buffer,
            self.dist_buffer,
        )
        return find_min(
            self.dist_buffer, dataset.n_timestep - s.length + 1, return_index
        )

    cdef Py_ssize_t transient_matches(
        self,
        SubsequenceView *v,
        Dataset dataset,
        Py_ssize_t index,
        double threshold,
        double **distances,
        Py_ssize_t **indicies,
    ) nogil:
        distances[0] = <double*> malloc(sizeof(double) * dataset.n_timestep - v.length + 1)
        indicies[0] = <Py_ssize_t*> malloc(sizeof(double) * dataset.n_timestep - v.length + 1)
        cumulative_mean_std(
            dataset.get_sample(index, v.dim),
            dataset.n_timestep,
            v.length,
            self.mean_x,
            self.std_x,
        )
        _mass_distance(
            dataset.get_sample(index, v.dim),
            dataset.n_timestep,
            dataset.get_sample(v.index, v.dim) + v.start,
            v.length,
            v.mean,
            v.std,
            self.mean_x,
            self.std_x,
            self.x_buffer,
            self.y_buffer,
            distances[0],
        )
        cdef Py_ssize_t i, j
        j = 0
        for i in range(dataset.n_timestep - v.length + 1):
            if distances[0][i] < threshold:
                distances[0][j] = distances[0][i]
                j += 1
        return j

    cdef Py_ssize_t persistent_matches(
        self,
        Subsequence *s,
        Dataset dataset,
        Py_ssize_t index,
        double threshold,
        double **distances,
        Py_ssize_t **indicies,
    ) nogil:
        distances[0] = <double*> malloc(sizeof(double) * dataset.n_timestep - s.length + 1)
        indicies[0] = <Py_ssize_t*> malloc(sizeof(double) * dataset.n_timestep - s.length + 1)
        cumulative_mean_std(
            dataset.get_sample(index, s.dim),
            dataset.n_timestep,
            s.length,
            self.mean_x,
            self.std_x,
        )
        _mass_distance(
            dataset.get_sample(index, s.dim),
            dataset.n_timestep,
            s.data,
            s.length,
            s.mean,
            s.std,
            self.mean_x,
            self.std_x,
            self.x_buffer,
            self.y_buffer,
            distances[0],
        )
        cdef Py_ssize_t i, j
        j = 0
        for i in range(dataset.n_timestep - s.length + 1):
            if distances[0][i] < threshold:
                distances[0][j] = distances[0][i]
                indicies[0][j] = i
                j += 1
        return j


cdef void _mass_distance(
    double *x,
    Py_ssize_t x_length,
    double *y,
    Py_ssize_t y_length,
    double mean,
    double std,
    double *mean_x,    # length x_length - y_length + 1
    double *std_x,     # length x_length - y_length + 1
    complex *y_buffer, # length x_length
    complex *x_buffer, # length x_length
    double *dist,      # length x_length - y_length + 1
) nogil:
    cdef Py_ssize_t i
    cdef double z
    for i in range(x_length):
        if i < y_length:
            y_buffer[i] = y[y_length - i - 1]
        else:
            y_buffer[i] = 0
        x_buffer[i] = x[i]

    _pocketfft.fft(y_buffer, x_length, 1.0)
    _pocketfft.fft(x_buffer, x_length, 1.0)
    for i in range(x_length):
        x_buffer[i] *= y_buffer[i]
    _pocketfft.ifft(x_buffer, x_length, 1.0 / x_length)

    for i in range(x_length - y_length + 1):
        if (
            std_x[i] <= EPSILON
            and not std <= EPSILON
            or std <= EPSILON
            and not std_x[i] <= EPSILON
        ):
            dist[i] = sqrt(y_length)
        elif std_x[i] <= EPSILON and std <= EPSILON:
            dist[i] = 0
        else:
            z = x_buffer[i + y_length - 1].real
            z = 2 * (y_length - (z - y_length * mean_x[i] * mean) / (std_x[i] * std))
            if z < EPSILON:
                dist[i] = 0
            else:
                dist[i] = sqrt(z)