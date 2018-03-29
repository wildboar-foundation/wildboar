# This file is part of pypf
#
# pypf is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pypf is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see
# <http://www.gnu.org/licenses/>.

# Authors: Isak Karlsson

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
    cdef readonly double unscaled_threshold
    cdef readonly Shapelet shapelet

    cdef readonly Node left
    cdef readonly Node right

    # if node_type == LEAF
    cdef double* distribution
    cdef size_t n_labels

cdef Node new_leaf_node(double* distribution, size_t n_labels)

cdef Node new_branch_node(SplitPoint sp,
                          Shapelet shapelet,
                          double unscaled_threshold)
