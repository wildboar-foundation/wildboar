cimport numpy as np

import numpy as np

from libc.math cimport INFINITY, NAN, sqrt
from libc.stdlib cimport free, malloc

from wildboar.utils._fft cimport _pocketfft
from wildboar.utils.data cimport Dataset
from wildboar.utils.stats cimport (
    IncStats,
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


cdef class ScaledMatrixProfileSubsequenceDistanceMeasure(ScaledSubsequenceDistanceMeasure):
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
        _cumulative_mean_std(
            dataset.get_sample(index, dim=s.dim),
            dataset.n_timestep,
            s.length,
            self.mean_x,
            self.std_x,
        )
        _mass_distance(
            dataset.get_sample(index, dim=s.dim),
            dataset.n_timestep,
            dataset.get_sample(s.index, dim=s.dim) + s.start,
            s.length,
            s.mean,
            s.std,
            self.mean_x,
            self.std_x,
            self.x_buffer,
            self.y_buffer,
            self.dist_buffer,
        )
        return _find_min(
            self.dist_buffer, dataset.n_timestep - s.length + 1, return_index
        )

    cdef double persistent_distance(
        self,
        Subsequence *s,
        Dataset dataset,
        Py_ssize_t index,
        Py_ssize_t *return_index=NULL,
    ) nogil:
        _cumulative_mean_std(
            dataset.get_sample(index, dim=s.dim),
            dataset.n_timestep,
            s.length,
            self.mean_x,
            self.std_x,
        )
        _mass_distance(
            dataset.get_sample(index, dim=s.dim),
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
        return _find_min(
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
        _cumulative_mean_std(
            dataset.get_sample(index, dim=v.dim),
            dataset.n_timestep,
            v.length,
            self.mean_x,
            self.std_x,
        )
        _mass_distance(
            dataset.get_sample(index, dim=v.dim),
            dataset.n_timestep,
            dataset.get_sample(v.index, dim=v.dim) + v.start,
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
        _cumulative_mean_std(
            dataset.get_sample(index, dim=s.dim),
            dataset.n_timestep,
            s.length,
            self.mean_x,
            self.std_x,
        )
        _mass_distance(
            dataset.get_sample(index, dim=s.dim),
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


cdef double _find_min(double *x, Py_ssize_t n, Py_ssize_t *min_index) nogil:
    cdef double min_val = INFINITY
    cdef Py_ssize_t i

    for i in range(n):
        if x[i] < min_val:
            min_index[0] = i
            min_val = x[i]

    return min_val


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
        z = x_buffer[i + y_length - 1].real
        z = 2 * (y_length - (z - y_length * mean_x[i] * mean) / (std_x[i] * std))
        if z < 0:
            dist[i] = 0
        else:
            dist[i] = sqrt(z)


cdef void _cumulative_mean_std(
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
            std = inc_stats_variance(&stats)
            if std == 0.0:
                std = 1.0
            else:
                std = sqrt(std)
            x_std[i - (y_length - 1)] = std 
            inc_stats_remove(&stats, 1.0, x[i - (y_length - 1)])

# cdef void _mass(
#     double *x,
#     Py_ssize_t x_length,
#     double *y,
#     Py_ssize_t y_length,
#     double *dist,
# ) nogil:
#     cdef Py_ssize_t i, j
#     cdef Py_ssize_t profile_length = x_length - y_length + 1
#     cdef complex *x_buffer = <complex*> malloc(sizeof(complex) * x_length)
#     cdef complex *y_buffer = <complex*> malloc(sizeof(complex) * x_length)
#     cdef double *x_mean = <double*> malloc(sizeof(double) * x_length - y_length + 1)
#     cdef double *x_std = <double*> malloc(sizeof(double) * x_length - y_length + 1)
#     cdef double std, y_std, y_mean
#     cdef IncStats stats
#     inc_stats_init(&stats)
#     for i in range(y_length):
#         inc_stats_add(&stats, 1.0, y[i])
    
#     y_std = sqrt(inc_stats_variance(&stats))
#     if y_std == 0:
#         y_std = 1.0
#     y_mean = stats.mean

#     _cumulative_mean_std(x, x_length, y_length, x_mean, x_std)

#     _mass(
#         x,
#         x_length,
#         y,
#         y_length,
#         y_mean, 
#         y_std,
#         x_mean,
#         x_std,
#         x_buffer,
#         y_buffer,
#         dist,
#     )
#     free(x_buffer)
#     free(y_buffer)
#     free(x_mean)
#     free(x_std)

# def test():
#     cdef np.ndarray x = np.random.RandomState(123).randn(100)
#     cdef np.ndarray y = np.array([2,2,2,2,3,3,3,10,10,2], dtype=np.double)
#     cdef np.ndarray d = np.zeros(100 - 10 + 1, dtype=np.double)
#     _mass_distance(<double*> x.data, 100, <double*> y.data, 10, <double*> d.data)
#     print("D", d)

# def test_dataset():
#     x = np.random.RandomState(123).randn(100).reshape(1, -1)
#     y = np.array([2,2,2,2,3,3,3,10,10,2], dtype=np.double)
#     cdef Subsequence s
#     cdef Dataset dataset = Dataset(x)
#     cdef SubsequenceDistanceMeasure distance_measure = ScaledMatrixProfileSubsequenceDistanceMeasure()
#     distance_measure.reset(dataset)
#     distance_measure.from_array(&s, (0, y))
#     cdef double *distances
#     cdef Py_ssize_t *indicies

#     n_matches = distance_measure.persistent_matches(&s, dataset, 0, 4, &distances, &indicies)

#     for i in range(n_matches):
#         print(indicies[i], distances[i])
