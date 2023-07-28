from cython cimport view

ctypedef const double[:, :, ::view.contiguous] TSArray

