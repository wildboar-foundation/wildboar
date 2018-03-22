cpdef double safe_info(double xw,
                       double[:] x,
                       double yw,
                       double[:] y,
                       double n_examples) nogil

cdef double info(double left_sum,
                 double* left_count,
                 double right_sum,
                 double* right_count,
                 size_t n_samples,
                 size_t n_labels) nogil
