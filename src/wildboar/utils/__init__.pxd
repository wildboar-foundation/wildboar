from cython cimport view

ctypedef const double[:, :, ::view.contiguous] TSArray
ctypedef const double[:, ::view.contiguous] TSSample
ctypedef const double[::view.contiguous] TSDimension