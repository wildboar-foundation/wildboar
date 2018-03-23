# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np

cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, INFINITY

cdef class Shapelet:
    """ Representing a shapelet carrying its own data.

    Note that the data is normalized during extraction, if using
    `shapelet_info_extract_shapelet`
    
    """

    def __cinit__(self, size_t length):
        self.data = <double*> malloc(sizeof(double) * length)
        self.length = length

    def __dealloc__(self):
        free(self.data)

    cpdef np.ndarray[np.float64_t] get_data(self):
        cdef np.ndarray[np.float64_t] arr = np.empty(self.length,
                                                     dtype=np.float64)
        cdef size_t i
        for i in range(self.length):
            arr[i] = self.data[i]
        return arr

    cdef double distance(self, const SlidingDistance t, size_t t_index) nogil:
        cdef size_t sample_offset = t_index * t.sample_stride
        cdef double current_value = 0
        cdef double mean = 0
        cdef double std = 0
        cdef double dist = 0
        cdef double min_dist = INFINITY

        cdef double ex = 0
        cdef double ex2 = 0

        cdef size_t i
        cdef size_t j
        cdef size_t buffer_pos
        
        for i in range(t.n_timestep):
            current_value = t.X[sample_offset + t.timestep_stride * i]
            ex += current_value
            ex2 += current_value * current_value

            buffer_pos = i % self.length
            t.X_buffer[buffer_pos] = current_value
            t.X_buffer[buffer_pos + self.length] = current_value
            if i >= self.length - 1:
                j = (i + 1) % self.length
                mean = ex / self.length
                std = sqrt(ex2 / self.length - mean * mean)
                dist = shapelet_subsequence_distance(self.length, # length of shapelet
                                                     self.data,   # normalized shapelet
                                                     j,           # buffer offset
                                                     mean,        # buffer mean
                                                     std,         # buffer std
                                                     t.X_buffer,  
                                                     min_dist)
                if dist < min_dist:
                    min_dist = dist

                current_value = t.X_buffer[j]
                ex -= current_value
                ex2 -= current_value * current_value

        return sqrt(min_dist)

cdef inline double shapelet_subsequence_distance(size_t length,
                                                 double* shapelet,
                                                 size_t j,
                                                 double mean,
                                                 double std,
                                                 double* X_buffer,
                                                 double min_dist) nogil:
    cdef double dist = 0
    cdef double x
    cdef size_t i
    cdef bint std_zero = std == 0
    for i in range(length):
        if dist >= min_dist:
            break
        
        x = shapelet[i]
        if not std_zero:
            x -= (X_buffer[i + j] - mean) / std
        dist += x * x
    
    return dist


cdef inline double shapelet_info_subsequence_distance(size_t offset,
                                                      size_t length,
                                                      double s_mean,
                                                      double s_std,
                                                      size_t j,
                                                      double mean,
                                                      double std,
                                                      double* X,
                                                      size_t timestep_stride,
                                                      double* X_buffer,
                                                      double min_dist) nogil:
    # Compute the distance between the shapelet (starting at `offset`
    # and ending at `offset + length` normalized with `s_mean` and
    # `s_std` with the shapelet in `X_buffer` starting at `0` and
    # ending at `length` normalized with `mean` and `std`
    cdef double dist = 0
    cdef double x
    cdef size_t i
    cdef bint std_zero = std == 0
    cdef bint s_std_zero = s_std == 0
    
    # distance is zero
    if s_std_zero and std_zero:
        return 0
    
    for i in range(length):
        if dist >= min_dist:
            break
        
        x = (X[offset + timestep_stride * i] - s_mean) / s_std
        if not std_zero:
            x -= (X_buffer[i + j] - mean) / std
        dist += x * x
    
    return dist


cdef Shapelet shapelet_info_extract_shapelet(ShapeletInfo s, const SlidingDistance t):
    """Extract (i.e., allocate) a shapelet to be stored outside the
    store. The `ShapeletInfo` is extpected to have `mean` and `std`
    computed.

    :param s: information about a shapelet
    :param t: the time series storage
    :return: a normalized shapelet
    """
    cdef Shapelet shapelet = Shapelet(s.length)
    cdef double* data = shapelet.data
    cdef size_t shapelet_offset = (s.index * t.sample_stride +
                                   s.start * t.timestep_stride)
    cdef size_t i
    if s.std == 0:
        for i in range(s.length):
            data[i] = 0.0
    else:
        for i in range(s.length):
            data[i] = (t.X[shapelet_offset + t.timestep_stride * i] - s.mean) / s.std

    return shapelet

cdef int shapelet_info_update_statistics(ShapeletInfo* s, const SlidingDistance t) nogil:
    cdef size_t shapelet_offset = (s.index * t.sample_stride +
                                   s.start * t.timestep_stride)
    cdef double ex = 0
    cdef double ex2 = 0
    cdef size_t i
    for i in range(s.length):
        current_value = t.X[shapelet_offset + i * t.timestep_stride]
        ex += current_value
        ex2 += current_value**2
        
    s[0].mean = ex / s.length
    s[0].std = sqrt(ex2 / s.length - s[0].mean * s[0].mean)
    return 0

cdef int shapelet_info_distances(ShapeletInfo s,
                                 const size_t* indicies,
                                 size_t n_indicies,
                                 const SlidingDistance t,
                                 double* result) nogil:
    cdef size_t p
    for p in range(n_indicies):
        result[p] = shapelet_info_distance(s, t, indicies[p])
    return 0

cdef double shapelet_info_distance(ShapeletInfo s, const SlidingDistance t, size_t t_index) nogil:
    cdef size_t sample_offset = t_index * t.sample_stride
    cdef size_t shapelet_offset = (s.index * t.sample_stride +
                                   s.start * t.timestep_stride)
    
    cdef double current_value = 0
    cdef double mean = 0
    cdef double std = 0
    cdef double dist = 0
    cdef double min_dist = INFINITY
    
    cdef double ex = 0
    cdef double ex2 = 0

    cdef size_t i
    cdef size_t j
    cdef size_t buffer_pos
        
    for i in range(t.n_timestep):
        current_value = t.X[sample_offset + t.timestep_stride * i]
        ex += current_value
        ex2 += current_value * current_value

        buffer_pos = i % s.length
        t.X_buffer[buffer_pos] = current_value
        t.X_buffer[buffer_pos + s.length] = current_value
        if i >= s.length - 1:
            j = (i + 1) % s.length
            mean = ex / s.length
            std = sqrt(ex2 / s.length - mean * mean)
            dist = shapelet_info_subsequence_distance(
                shapelet_offset,
                s.length,
                s.mean,
                s.std,
                j,
                mean,
                std,
                t.X,
                t.timestep_stride,
                t.X_buffer,
                min_dist)
                
            if dist < min_dist:
                min_dist = dist

            current_value = t.X_buffer[j]
            ex -= current_value
            ex2 -= current_value * current_value

    return sqrt(min_dist)


cdef SlidingDistance new_sliding_distance(np.ndarray[np.float64_t, ndim=2, mode="c"] X):
    cdef SlidingDistance sd
    sd.n_samples = X.shape[0]
    sd.n_timestep = X.shape[1]
    sd.X = <double*> X.data
    sd.sample_stride = <size_t> X.strides[0] / <size_t> X.itemsize
    sd.timestep_stride = <size_t> X.strides[1] / <size_t> X.itemsize
    sd.X_buffer = <double*> malloc(sizeof(double) * 2 * sd.n_timestep)
    return sd

cdef int free_sliding_distance(SlidingDistance sd) nogil:
    free(sd.X_buffer)
    sd.X_buffer = NULL
    # sd.X is freed by its owner
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int sliding_distance(double[:] s,
                           double[:, :] X,
                           long[:] idx,
                           double[:] out) nogil except -1:
    cdef Py_ssize_t i, j
    cdef Py_ssize_t m = idx.shape[0]
    cdef Py_ssize_t n = X.shape[1]
    cdef double* buf = <double*>malloc(n * 2 * sizeof(double))
    if not buf:
        return -1
    try:
        for i in range(m):
            j = idx[i]
            out[i] = sliding_distance_(s, X, j, buf)
        return 0
    finally:
        free(buf)


cpdef sliding_distance_one(double[:] s, double[:, :] X, Py_ssize_t i):
    cdef Py_ssize_t n = X.shape[1]
    cdef double* buf = <double*>malloc(n * 2 * sizeof(double))
    if not buf:
        raise MemoryError()
    cdef double dist = sliding_distance_(s, X, i, buf)
    try:
        return dist
    except:
        free(buf)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double sliding_distance_(double[:] s, double[:,:] X, Py_ssize_t
                              ts, double* buf) nogil:
    cdef Py_ssize_t m = s.shape[0]
    cdef Py_ssize_t n = X.shape[1]
    cdef double d = 0
    cdef double mean = 0
    cdef double sigma = 0
    cdef double dist = 0
    cdef double min_dist = INFINITY

    cdef double ex = 0
    cdef double ex2 = 0
    cdef Py_ssize_t i, j
    for i in range(n):
        d = X[ts, i]
        ex += d
        ex2 += (d * d)
        buf[i % m] = d
        buf[(i % m) + m] = d
        if i >= m - 1:
            j = (i + 1) % m
            mean = ex / m
            sigma = sqrt((ex2 / m) - (mean * mean))
            dist = distance(s, buf, j, m, mean, sigma, min_dist)
            if dist < min_dist:
                min_dist = dist
            ex -= buf[j]
            ex2 -= (buf[j] * buf[j])
    return sqrt(min_dist / m)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double distance(double[:] s,
                     double* buf,
                     Py_ssize_t j,
                     Py_ssize_t m,
                     double mean,
                     double std,
                     double bsf) nogil:
    cdef double sf = 0
    cdef double x = 0
    cdef Py_ssize_t i
    for i in range(m):
        if sf >= bsf:
            break
        if std == 0:
            x = s[i]
        else:
            x = (buf[i + j] - mean) / std - s[i]
        sf += x * x
    return sf
