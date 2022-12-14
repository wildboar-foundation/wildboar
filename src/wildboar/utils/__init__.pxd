from cython cimport view

ctypedef const double[:, :, ::view.contiguous] TSArray

from .utils._misc cimport List