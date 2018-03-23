cimport numpy as np
from libc.math import NaN
from pypf._sliding_distance cimport ShapeletInfo, Shapelet

cdef struct SplitPoint:
   size_t split_point
   double threshold
   ShapeletInfo shapelet_info


cdef SplitPoint new_split_point(size_t split_point,
                                double threshold,
                                ShapeletInfo shapelet_info) nogil

cdef class Node:
    cdef readonly bint is_leaf

    # if node_type == BRANCH
    cdef readonly double threshold
    cdef readonly Shapelet shapelet

    cdef readonly Node left
    cdef readonly Node right

    # if node_type == LEAF
    cdef double* distribution
    cdef size_t n_labels

    cpdef np.ndarray[np.float64_t] get_proba(self)

cdef Node new_leaf_node(double* distribution, size_t n_labels)

cdef Node new_branch_node(SplitPoint sp, Shapelet shapelet)
