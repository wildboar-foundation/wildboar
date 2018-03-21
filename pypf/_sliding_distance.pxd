cdef class SlidingDistance:
   cdef size_t n_samples
   cdef size_t n_timestep

   cdef double* X
   cdef size_t sample_stride
   cdef size_t timestep_stride

   cdef double* X_buffer # buffer for normalization

   cdef int shapelet_statistics(self,
                                size_t s_index,
                                size_t s_start,
                                size_t s_len,
                                double* mean,
                                double* std) nogil
   
   cdef int distance_list(self,
                          size_t s_index,
                          size_t s_start,
                          size_t s_len,
                          const size_t* indicies,
                          size_t n_indicies,
                          double* result)

   cdef double distance(self,
                        size_t s_index,
                        size_t s_start,
                        size_t s_len,
                        size_t t_index)

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
