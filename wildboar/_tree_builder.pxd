# cython: language_level=3

# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wildboar is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# Authors: Isak Samsten
cimport numpy as np

from ._distance cimport DistanceMeasure
from ._distance cimport ShapeletInfo, Shapelet

# TODO: include impurity score...
cdef struct SplitPoint:
    size_t split_point
    double threshold
    ShapeletInfo shapelet_info

cdef class Tree:
    cdef DistanceMeasure distance_measure

    cdef size_t _max_depth
    cdef size_t _capacity
    cdef size_t _n_labels  # 1 for regression

    cdef size_t _node_count
    cdef int *_left
    cdef int *_right
    cdef Shapelet ** _shapelets
    cdef double *_thresholds
    cdef double *_impurity
    cdef double *_values
    cdef double *_n_weighted_node_samples
    cdef size_t *_n_node_samples

    cdef int _increase_capacity(self) nogil except -1

    cdef int add_leaf_node(self, int parent, bint is_left, size_t n_node_samples, double n_weighted_node_samples) nogil

    cdef void set_leaf_value(self, size_t node_id, size_t out_label, double out_value) nogil

    cdef int add_branch_node(self, int parent, bint is_left, size_t n_node_samples, double n_weighted_node_samples,
                             Shapelet *shapelet, double threshold, double impurity) nogil

    cpdef np.ndarray predict(self, object X)

    cpdef np.ndarray apply(self, object X)

    cpdef np.ndarray decision_path(self, object X)
