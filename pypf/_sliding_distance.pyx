# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from __future__ import print_function
import numpy as np
cimport numpy as np

cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, INFINITY

cdef class SlidingDistance:

    def __cinit__(self, np.ndarray[np.float64_t, ndim=2, mode="c"] X):
        self.n_samples = X.shape[0]
        self.n_timestep = X.shape[1]
        self.X = <double*> X.data
        self.sample_stride = <size_t> X.strides[0] / <size_t> X.itemsize
        self.timestep_stride = <size_t> X.strides[1] / <size_t> X.itemsize

        self.X_buffer = <double*> malloc(sizeof(double) * 2 * self.n_timestep)
        

    def __dealloc__(self):
        free(self.X_buffer)

    cdef int shapelet_statistics(self,
                                 size_t s_index,
                                 size_t s_start,
                                 size_t s_len,
                                 double* mean,
                                 double* std) nogil:
        cdef size_t shapelet_offset = (s_index * self.sample_stride +
                                       s_start * self.timestep_stride)
        cdef double ex = 0
        cdef double ex2 = 0
        cdef size_t i
        for i in range(s_len):
            current_value = self.X[shapelet_offset + i * self.timestep_stride]
            ex += current_value
            ex2 += current_value**2
            
        mean[0] = ex / s_len
        std[0] = sqrt(ex2 / s_len - mean[0] * mean[0])
        return 0
    

    cdef int distance_list(self,
                           size_t s_index,
                           size_t s_start,
                           size_t s_len,
                           const size_t* indicies,
                           size_t n_indicies,
                           double* result):

        # TODO: compute mean and std of shapelet outside of this function
        cdef size_t p
        for p in range(n_indicies):
            result[p] = self.distance(s_index, s_start, s_len, p)

        return 0

    cdef double distance(self,
                         size_t s_index,
                         size_t s_start,
                         size_t s_len,
                         size_t t_index):

        cdef size_t sample_offset = t_index * self.sample_stride
        cdef size_t shapelet_offset = (s_index * self.sample_stride +
                                       s_start * self.timestep_stride)
        
        cdef double current_value = 0
        cdef double mean = 0
        cdef double std = 0
        cdef double dist = 0
        cdef double min_dist = INFINITY

        cdef double s_mean
        cdef double s_std
        
        cdef double ex = 0
        cdef double ex2 = 0

        cdef size_t i
        cdef size_t j
        cdef size_t buffer_pos

        # compute mean outside
        self.shapelet_statistics(s_index, s_start, s_len, &s_mean, &s_std)

        print(sample_offset, s_mean, s_std)
        
        for i in range(self.n_timestep):
            current_value = self.X[sample_offset + self.timestep_stride * i]
            ex += current_value
            ex2 += current_value * current_value

            buffer_pos = i % s_len
            self.X_buffer[buffer_pos] = current_value
            self.X_buffer[buffer_pos + s_len] = current_value
            if i >= s_len - 1:
                j = (i + 1) % s_len
                mean = ex / s_len
                std = sqrt(ex2 / s_len - mean * mean)
                dist = self.subsequence_distance(shapelet_offset,
                                                 s_len, s_mean, s_std,
                                                 j, mean, std,
                                                 min_dist)
                if dist < min_dist:
                    min_dist = dist

                current_value = self.X_buffer[j]
                ex -= current_value
                ex2 -= current_value * current_value

        return sqrt(min_dist)


    cdef double subsequence_distance(self,
                                     size_t offset,
                                     size_t length,
                                     double s_mean,
                                     double s_std,
                                     size_t j,
                                     double mean,
                                     double std,
                                     double min_dist) nogil:
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
            
            x = (self.X[offset + self.timestep_stride * i] - s_mean) / std
            if not std_zero:
                x -= (self.X_buffer[i + j] - mean) / std
            dist += x * x

        return dist
    
       

                  

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

cdef double sliding_distance_ptr(size_t s_index, size_t s_start,
                                 size_t s_len, double* X, size_t
                                 X_sample_stride, size_t
                                 X_time_stride, size_t t_index,
                                 double* buf) nogil:
    # cdef Py_ssize_t m = s.shape[0]
    # cdef Py_ssize_t n = X.shape[1]
    # cdef double d = 0
    # cdef double mean = 0
    # cdef double sigma = 0
    # cdef double dist = 0
    # cdef double min_dist = INFINITY

    # cdef double ex = 0
    # cdef double ex2 = 0
    # cdef Py_ssize_t i, j
    # for i in range(n):
    #     d = X[ts, i]
    #     ex += d
    #     ex2 += (d * d)
    #     buf[i % m] = d
    #     buf[(i % m) + m] = d
    #     if i >= m - 1:
    #         j = (i + 1) % m
    #         mean = ex / m
    #         sigma = sqrt((ex2 / m) - (mean * mean))
    #         dist = distance(s, buf, j, m, mean, sigma, min_dist)
    #         if dist < min_dist:
    #             min_dist = dist
    #         ex -= buf[j]
    #         ex2 -= (buf[j] * buf[j])
    # return sqrt(min_dist / m)
    return 0.0


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
