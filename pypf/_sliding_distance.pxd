cimport numpy as np

cdef class Shapelet:
   cdef readonly size_t length
   cdef double* data

   cdef double distance(self, const SlidingDistance t, size_t t_index) nogil

   cpdef np.ndarray[np.float64_t] get_data(self)


cdef double shapelet_subsequence_distance(size_t length,
                                          double* shapelet,
                                          size_t j,
                                          double mean,
                                          double std,
                                          double* X,
                                          size_t timestep_stride,
                                          double* X_buffer,
                                          double min_dist) nogil


cdef struct ShapeletInfo:
   size_t index
   size_t start
   size_t length
   double mean
   double std

cdef double shapelet_info_subsequence_distance(size_t offset,
                                               size_t length,
                                               double s_mean,
                                               double s_std,
                                               size_t j,
                                               double mean,
                                               double std,
                                               double* X,
                                               size_t timestep_stride,
                                               double* X_buffer,
                                               double min_dist) nogil



cdef int shapelet_info_update_statistics(ShapeletInfo* s,
                                         const SlidingDistance t) nogil

cdef int shapelet_info_distances(ShapeletInfo s,
                                 const size_t* samples,
                                 size_t n_samples,
                                 const SlidingDistance t,
                                 double* result) nogil

cdef double shapelet_info_distance(ShapeletInfo s, const SlidingDistance t, size_t t_index) nogil

cdef Shapelet shapelet_info_extract_shapelet(ShapeletInfo s, const SlidingDistance t)


cdef SlidingDistance new_sliding_distance(np.ndarray[np.float64_t, ndim=2, mode="c"] X)

cdef int free_sliding_distance(SlidingDistance sd) nogil


cdef struct SlidingDistance:
   size_t n_samples
   size_t n_timestep

   double* X
   size_t sample_stride
   size_t timestep_stride

   double* X_buffer # buffer for normalization


cpdef int sliding_distance(double[:] s,
                           double[:, :] X,
                           long[:] idx,
                           double[:] out) nogil except -1
