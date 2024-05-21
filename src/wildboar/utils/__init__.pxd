# cython: language_level=3
from cython cimport view

ctypedef const double[:, :, ::view.contiguous] TSArray

