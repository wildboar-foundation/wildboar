#Authors: Isak Karlsson

cdef double info(double left_sum,
                 double* left_count,
                 double right_sum,
                 double* right_count,
                 size_t n_labels) nogil
