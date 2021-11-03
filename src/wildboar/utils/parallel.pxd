from .data cimport Dataset

cdef class ForeachSample:

    cdef Dataset x_in

    cdef void work(self, Py_ssize_t i) nogil

cdef class MapSample(ForeachSample):

    cdef double[:, :] result

    cdef double map(self, double *sample) nogil