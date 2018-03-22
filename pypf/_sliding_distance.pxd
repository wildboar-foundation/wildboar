cdef struct Shapelet:
   size_t index
   size_t start
   size_t length
   double mean
   double std


cdef class SlidingDistance:
   cdef size_t n_samples
   cdef size_t n_timestep

   cdef double* X
   cdef size_t sample_stride
   cdef size_t timestep_stride

   cdef double* X_buffer # buffer for normalization

   cdef int shapelet_statistics(self, Shapelet* shapelet) nogil
   
   cdef int distance_list(self,
                          Shapelet shapelet,
                          const size_t* indicies,
                          size_t n_indicies,
                          double* result) nogil

   cdef double distance(self, Shapelet shapelet, size_t t_index) nogil

   cdef double subsequence_distance(self,
                                    size_t offset,
                                    size_t length,
                                    double s_mean,
                                    double s_std,
                                    size_t j,
                                    double mean,
                                    double std,
                                    double min_dist) nogil
                        

cpdef int sliding_distance(double[:] s,
                           double[:, :] X,
                           long[:] idx,
                           double[:] out) nogil except -1
