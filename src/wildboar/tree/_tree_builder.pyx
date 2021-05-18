# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
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

import numpy as np
cimport numpy as np

from libc.math cimport INFINITY
from libc.math cimport NAN

from libc.stdlib cimport malloc
from libc.stdlib cimport free

from libc.string cimport memcpy
from libc.string cimport memset

from ..distance._distance cimport TSDatabase
from ..distance._distance cimport DistanceMeasure

from ..distance._distance cimport TSView
from ..distance._distance cimport TSCopy

from ..distance._distance cimport ts_database_new

from ..distance._distance cimport ts_view_init
from ..distance._distance cimport ts_view_free
from ..distance._distance cimport ts_copy_free
from ..distance._distance cimport new_distance_measure

from ._impurity cimport entropy

from .._utils cimport label_distribution
from .._utils cimport argsort
from .._utils cimport rand_int
from .._utils cimport RAND_R_MAX
from .._utils cimport RollingVariance
from .._utils cimport safe_realloc
from .._utils cimport rand_uniform


cpdef Tree _make_tree(
    DistanceMeasure distance_measure,
    Py_ssize_t n_labels,
    list shapelets,
    np.ndarray threshold,
    np.ndarray value,
    np.ndarray left,
    np.ndarray right,
    np.ndarray impurity,
    np.ndarray n_node_samples,
    np.ndarray n_weighted_node_samples
):
    cdef Tree tree = Tree(distance_measure, n_labels, capacity=len(shapelets) + 1)
    cdef Py_ssize_t node_count = len(shapelets)
    cdef Py_ssize_t i
    cdef Py_ssize_t dim
    cdef np.ndarray arr
    cdef TSCopy *shapelet
    cdef np.ndarray value_reshape = value.reshape(-1)

    tree._node_count = node_count
    for i in range(node_count):
        if shapelets[i] is not None:
            dim, arr = shapelets[i]
            shapelet = <TSCopy*> malloc(sizeof(TSCopy))
            distance_measure.init_ts_copy_from_ndarray(shapelet, arr, dim)
            tree._shapelets[i] = shapelet
        else:
            tree._shapelets[i] = NULL
        tree._thresholds[i] = threshold[i]
        tree._left[i] = left[i]
        tree._right[i] = right[i]
        tree._impurity[i] = impurity[i]
        tree._n_node_samples[i] = n_node_samples[i]
        tree._n_weighted_node_samples[i] = n_weighted_node_samples[i]

    for i in range(node_count * n_labels):
        tree._values[i] = value_reshape[i]
    return tree


cdef class Tree:
    def __cinit__(
        self,
        DistanceMeasure distance_measure,
        Py_ssize_t n_labels,
        Py_ssize_t capacity=10
    ):
        self.distance_measure = distance_measure
        self._node_count = 0
        self._capacity = capacity
        self._n_labels = n_labels
        self._shapelets = <TSCopy**> malloc(self._capacity * sizeof(TSCopy*))
        self._thresholds = <double*> malloc(self._capacity * sizeof(double))
        self._values = <double*> malloc(self._capacity * self._n_labels * sizeof(double))
        self._left = <Py_ssize_t*> malloc(self._capacity * sizeof(Py_ssize_t))
        self._right = <Py_ssize_t*> malloc(self._capacity * sizeof(Py_ssize_t))
        self._impurity = <double*> malloc(self._capacity * sizeof(double))
        self._n_node_samples = <Py_ssize_t*> malloc(self._capacity * sizeof(Py_ssize_t))
        self._n_weighted_node_samples = <double*> malloc(self._capacity * sizeof(double))

    def __dealloc__(self):
        cdef Py_ssize_t i
        if self._shapelets != NULL:
            for i in range(self._node_count):
                if self._shapelets[i] != NULL:
                    free(self._shapelets[i])
                    ts_copy_free(self._shapelets[i])
            free(self._shapelets)

        if self._thresholds != NULL:
            free(self._thresholds)

        if self._values != NULL:
            free(self._values)

        if self._left != NULL:
            free(self._left)

        if self._right != NULL:
            free(self._right)

        if self._impurity != NULL:
            free(self._impurity)

        if self._n_node_samples != NULL:
            free(self._n_node_samples)

        if self._n_weighted_node_samples != NULL:
            free(self._n_weighted_node_samples)

    def __reduce__(self):
        return _make_tree, (
            self.distance_measure,
            self._n_labels,
            self.shapelet,
            self.threshold,
            self.value,
            self.left,
            self.right,
            self.impurity,
            self.n_node_samples,
            self.n_weighted_node_samples,
        )


    @property
    def max_depth(self):
        return self._max_depth

    @property
    def value(self):
        cdef np.ndarray arr = np.empty(self._node_count * self._n_labels, dtype=np.float64)
        cdef Py_ssize_t i
        for i in range(self._n_labels * self._node_count):
            arr[i] = self._values[i]
        return arr.reshape(self._node_count, self._n_labels)

    @property
    def shapelet(self):
        cdef TSCopy *shapelet
        cdef np.ndarray temp
        cdef Py_ssize_t i, j
        cdef list ret = []
        for i in range(self._node_count):
            shapelet = self._shapelets[i]
            if shapelet != NULL:
                temp = np.empty(shapelet[0].length, dtype=np.float64)
                for j in range(shapelet[0].length):
                    temp[j] = shapelet[0].data[j]
                ret.append((shapelet[0].dim, temp))
            else:
                ret.append(None)
        return ret

    @property
    def n_node_samples(self):
        cdef np.ndarray arr = np.zeros(self._node_count, dtype=np.intp)
        cdef Py_ssize_t i
        for i in range(self._node_count):
            arr[i] = self._n_node_samples[i]
        return arr

    @property
    def n_weighted_node_samples(self):
        cdef np.ndarray arr = np.zeros(self._node_count, dtype=np.float64)
        cdef Py_ssize_t i
        for i in range(self._node_count):
            arr[i] = self._n_weighted_node_samples[i]
        return arr

    @property
    def left(self):
        cdef np.ndarray arr = np.empty(self._node_count, dtype=np.intp)
        cdef Py_ssize_t i
        for i in range(self._node_count):
            arr[i] = self._left[i]
        return arr

    @property
    def right(self):
        cdef np.ndarray arr = np.empty(self._node_count, dtype=np.intp)
        cdef Py_ssize_t i
        for i in range(self._node_count):
            arr[i] = self._right[i]
        return arr

    @property
    def threshold(self):
        cdef np.ndarray arr = np.empty(self._node_count, dtype=np.float64)
        cdef Py_ssize_t i
        for i in range(self._node_count):
            arr[i] = self._thresholds[i]
        return arr

    @property
    def impurity(self):
        cdef np.ndarray arr = np.empty(self._node_count, dtype=np.float64)
        cdef Py_ssize_t i
        for i in range(self._node_count):
            arr[i] = self._impurity[i]
        return arr

    cpdef np.ndarray predict(self, object X):
        cdef np.ndarray apply = self.apply(X)
        cdef np.ndarray predict = np.take(self.value, apply, axis=0, mode="clip")
        if self._n_labels == 1:
            predict = predict.reshape(X.shape[0])
        return predict

    cpdef np.ndarray apply(self, object X):
        if not isinstance(X, np.ndarray):
            raise ValueError(f"X should be np.ndarray, got {type(X)}")

        cdef TSDatabase ts = ts_database_new(X)
        cdef np.ndarray[np.npy_intp] out = np.zeros((ts.n_samples,), dtype=np.intp)
        cdef Py_ssize_t *out_data = <Py_ssize_t*> out.data
        cdef TSCopy *shapelet
        cdef double threshold
        cdef Py_ssize_t node_index
        cdef Py_ssize_t i
        with nogil:
            for i in range(ts.n_samples):
                node_index = 0
                while self._left[node_index] != -1:
                    threshold = self._thresholds[node_index]
                    shapelet = self._shapelets[node_index]
                    if self.distance_measure.ts_copy_sub_distance(shapelet, &ts, i) <= threshold:
                        node_index = self._left[node_index]
                    else:
                        node_index = self._right[node_index]
                out_data[i] = <Py_ssize_t> node_index
        return out

    cpdef np.ndarray decision_path(self, object X):
        if not isinstance(X, np.ndarray):
            raise ValueError(f"X should be np.ndarray, got {type(X)}")
        cdef TSDatabase ts = ts_database_new(X)
        cdef np.ndarray out = np.zeros((ts.n_samples, self.node_count), order="c", dtype=np.intp)

        cdef Py_ssize_t *out_data = <Py_ssize_t*> out.data
        cdef Py_ssize_t i_stride = <Py_ssize_t> out.strides[0] / <Py_ssize_t> out.itemsize
        cdef Py_ssize_t n_stride = <Py_ssize_t> out.strides[1] / <Py_ssize_t> out.itemsize
        cdef Py_ssize_t node_index
        cdef Py_ssize_t i
        cdef TSCopy *shapelet
        cdef double threshold
        with nogil:
            for i in range(ts.n_samples):
                node_index = 0
                while self._left[node_index] != -1:
                    out_data[i * i_stride + node_index * n_stride] = 1
                    # out_data[i, node_index] = 1
                    threshold = self._thresholds[node_index]
                    shapelet = self._shapelets[node_index]
                    if self.distance_measure.ts_copy_sub_distance(shapelet, &ts, i) <= threshold:
                        node_index = self._left[node_index]
                    else:
                        node_index = self._right[node_index]
        return out

    @property
    def node_count(self):
        return self._node_count

    cdef Py_ssize_t add_leaf_node(
        self,
        Py_ssize_t parent,
        bint is_left,
        Py_ssize_t n_node_samples,
        double n_weighted_node_samples,
    ) nogil:
        cdef Py_ssize_t node_id = self._node_count
        if node_id >= self._capacity:
            if self._increase_capacity() == -1:
                return -1

        self._n_node_samples[node_id] = n_node_samples
        self._n_weighted_node_samples[node_id] = n_weighted_node_samples
        if parent != -1:
            if is_left:
                self._left[parent] = node_id
            else:
                self._right[parent] = node_id
        self._left[node_id] = -1
        self._right[node_id] = -1
        self._impurity[node_id] = -1
        self._shapelets[node_id] = NULL
        self._node_count += 1
        return node_id

    cdef void set_leaf_value(
        self,
        Py_ssize_t node_id,
        Py_ssize_t out_label,
        double out_value,
    ) nogil:
        self._values[out_label + node_id * self._n_labels] = out_value

    cdef Py_ssize_t add_branch_node(
        self,
        Py_ssize_t parent,
        bint is_left,
        Py_ssize_t n_node_samples,
        double n_weighted_node_samples,
        TSCopy *shapelet,
        double threshold,
        double impurity,
    ) nogil:
        cdef Py_ssize_t node_id = self._node_count
        if node_id >= self._capacity:
            if self._increase_capacity() == -1:
                return -1

        self._impurity[node_id] = impurity
        self._n_node_samples[node_id] = n_node_samples
        self._n_weighted_node_samples[node_id] = n_weighted_node_samples
        self._thresholds[node_id] = threshold
        self._shapelets[node_id] = shapelet
        if parent != -1:
            if is_left:
                self._left[parent] = node_id
            else:
                self._right[parent] = node_id

        self._node_count += 1
        return node_id

    cdef Py_ssize_t _increase_capacity(self) nogil except -1:
        cdef Py_ssize_t new_capacity = self._node_count * 2
        cdef Py_ssize_t ret
        ret = safe_realloc(<void**> &self._shapelets, sizeof(TSCopy) * new_capacity)
        if ret == -1:
            return -1

        ret = safe_realloc(<void**> &self._thresholds, sizeof(double) * new_capacity)
        if ret == -1:
            return -1

        ret = safe_realloc(<void**> &self._impurity, sizeof(double) * new_capacity)
        if ret == -1:
            return -1

        ret = safe_realloc(<void**> &self._n_node_samples, sizeof(double) * new_capacity)
        if ret == -1:
            return -1

        ret = safe_realloc(<void**> &self._n_weighted_node_samples, sizeof(Py_ssize_t) * new_capacity)
        if ret == -1:
            return -1

        ret = safe_realloc(<void**> &self._values, sizeof(double) * self._n_labels * new_capacity)
        if ret == -1:
            return -1

        ret = safe_realloc(<void**> &self._left, sizeof(Py_ssize_t) * new_capacity)
        if ret == -1:
            return -1

        ret = safe_realloc(<void**> &self._right, sizeof(Py_ssize_t) * new_capacity)
        if ret == -1:
            return -1

        return 0


cdef inline SplitPoint new_split_point(
    Py_ssize_t split_point,
    double threshold,
    TSView shapelet_info,
) nogil:
    cdef SplitPoint s
    s.split_point = split_point
    s.threshold = threshold
    s.shapelet_info = shapelet_info
    return s


cdef class ShapeletTreeBuilder:
    cdef size_t random_seed

    cdef Py_ssize_t n_shapelets
    cdef Py_ssize_t min_shapelet_size
    cdef Py_ssize_t max_shapelet_size
    cdef Py_ssize_t max_depth
    cdef Py_ssize_t min_sample_split
    cdef Py_ssize_t current_node_id

    # the stride of the label array
    cdef Py_ssize_t label_stride

    # the weight of the j:th sample
    cdef double *sample_weights

    # the dataset of time series
    cdef TSDatabase td

    # the number of samples with non-zero weight
    cdef Py_ssize_t n_samples

    # buffer of samples from 0, ..., n_samples
    cdef Py_ssize_t *samples

    # temporary buffer of samples from 0, ..., n_samples
    cdef Py_ssize_t *samples_buffer

    # the sum of samples_weights
    cdef double n_weighted_samples

    # temporary buffer for distance computations
    cdef double *distance_buffer

    # the distance measure implementation
    cdef DistanceMeasure distance_measure

    # the tree structure representation
    cdef Tree tree

    def __cinit__(
        self,
        Py_ssize_t n_shapelets,
        Py_ssize_t min_shapelet_size,
        Py_ssize_t max_shapelet_size,
        Py_ssize_t max_depth,
        Py_ssize_t min_sample_split,
        object metric,
        dict metric_params,
        np.ndarray X,
        np.ndarray y,
        np.ndarray sample_weights,
        object random_state,
        *args,
        **kwargs
    ):
        self.random_seed = random_state.randint(0, RAND_R_MAX)
        self.n_shapelets = n_shapelets
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.current_node_id = 0

        self.td = ts_database_new(X)
        self.distance_measure = new_distance_measure(&self.td, metric, metric_params)
        self.label_stride = <Py_ssize_t> y.strides[0] / <Py_ssize_t> y.itemsize

        self.n_samples = X.shape[0]
        self.samples = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * self.n_samples)
        self.samples_buffer = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * self.n_samples)
        self.distance_buffer = <double*> malloc(sizeof(double) * self.n_samples)

        if (
            self.samples == NULL or
            self.distance_buffer == NULL or
            self.samples_buffer == NULL
        ):
            raise MemoryError()

        cdef Py_ssize_t i
        cdef Py_ssize_t j = 0
        for i in range(self.n_samples):
            if sample_weights is None or sample_weights[i] != 0.0:
                self.samples[j] = i
                j += 1

        self.n_samples = j
        self.n_weighted_samples = 0

        if sample_weights is None:
            self.sample_weights = NULL
        else:
            self.sample_weights = <double*> sample_weights.data

    def __dealloc__(self):
        free(self.samples)
        free(self.samples_buffer)
        free(self.distance_buffer)

    @property
    def tree_(self):
        return self.tree

    cpdef Py_ssize_t build_tree(self):
        cdef Py_ssize_t root_node_id
        cdef Py_ssize_t max_depth = 0
        with nogil:
            root_node_id = self._build_tree(0, self.n_samples, 0, -1, False, &max_depth)

        self.tree._max_depth = max_depth
        return root_node_id

    cdef Py_ssize_t new_leaf_node(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        Py_ssize_t parent,
        bint is_left,
    ) nogil:
        pass

    cdef Py_ssize_t new_branch_node(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        SplitPoint sp,
        TSCopy *shapelet,
        Py_ssize_t parent,
        bint is_left,
    ) nogil:
        cdef Py_ssize_t node_id
        node_id = self.tree.add_branch_node(
            parent,
            is_left,
            end - start,
            self.n_weighted_samples,
            shapelet,
            sp.threshold, -1,
        )
        return node_id


    cdef bint _is_pre_pruned(self, Py_ssize_t start, Py_ssize_t end) nogil:
        """Check if the tree should be pruned based on the samples in the
        region indicated by `start` and `end`

        Implementation detail: For optimization, subclasses *must* set
        the variable `self.n_weighted_samples` to the number of samples
        reaching this node in this method.
       
        Parameters
        ----------
        start : int
            The start position in samples
            
        end : int
            The end position in samples
        """
        pass

    cdef Py_ssize_t _build_tree(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        Py_ssize_t depth,
        Py_ssize_t parent,
        bint is_left,
        Py_ssize_t *max_depth,
    ) nogil:
        """Recursive function for building the tree

        Each call to this function is allowed to access and transpose
        samples (in `self.samples`) in the region indicated by `start`
        and `end`.

        Parameters
        ----------
        start : int
            The start position in samples
            
        end : int
            The end position in samples
            
        depth : int (out)
            The current recusion depth
            
        parent : int
            The index of the parent node
            
        is_left : bool
            True if current node is left
            
        max_depth : int (out)
            The max depth reached in the tree        
        """
        if depth > max_depth[0]:
            max_depth[0] = depth

        if self._is_pre_pruned(start, end) or depth >= self.max_depth:
            return self.new_leaf_node(start, end, parent, is_left)

        cdef SplitPoint split = self._split(start, end)
        cdef TSCopy *shapelet
        cdef Py_ssize_t current_node_id, left_node_id, right_node_id
        cdef Py_ssize_t err
        if split.split_point > start and end - split.split_point > 0:
            shapelet = <TSCopy*> malloc(sizeof(TSCopy))
            err = self.distance_measure.init_ts_copy(
                shapelet, &split.shapelet_info, &self.td
            )
            if err == -1:
                return -1

            # `shapelet` will be stored and freed by `self.tree`
            current_node_id = self.new_branch_node(
                start, end, split, shapelet, parent, is_left
            )
            left_node_id = self._build_tree(
                start, split.split_point, depth + 1, current_node_id, True, max_depth
            )
            right_node_id = self._build_tree(
                split.split_point, end, depth + 1, current_node_id, False, max_depth
            )

            ts_view_free(&split.shapelet_info)  # RECLAIM THIS MEMORY
            return current_node_id
        else:
            with gil:
                print("warn: split point outside allowed range. This is a bug. Please report me.")
            return self.new_leaf_node(start, end, parent, is_left)

    cdef SplitPoint _split(self, Py_ssize_t start, Py_ssize_t end) nogil:
        """Split the `sample` array indicated by the region specified by
        `start` and `end` by minimizing an impurity measure

        Parameters
        ----------
        start : int
            The start position in samples
            
        end : int
            The end position in samples
            
        Notes
        -----
         - `self._partition_distance_buffer`: implements the decision 
            to determine the split quality minimizing `impurity`
        """
        cdef Py_ssize_t split_point, best_split_point
        cdef double threshold, best_threshold
        cdef double impurity
        cdef double best_impurity
        cdef TSView shapelet
        cdef TSView best_shapelet
        cdef Py_ssize_t i

        ts_view_init(&best_shapelet)
        best_impurity = INFINITY
        best_threshold = NAN
        best_split_point = 0
        split_point = 0

        for i in range(self.n_shapelets):
            self._sample_shapelet(&shapelet, start, end)
            self.distance_measure.ts_view_sub_distances(
                &shapelet,
                &self.td,
                self.samples + start,
                self.distance_buffer + start,
                end - start,
            )
            argsort(self.distance_buffer + start, self.samples + start, end - start)
            self._partition_distance_buffer(
                start, end, &split_point, &threshold, &impurity
            )
            if impurity < best_impurity:
                # store the order of samples in `sample_buffer`
                memcpy(
                    self.samples_buffer,
                    self.samples + start,
                    sizeof(Py_ssize_t) * (end - start),
                )
                best_impurity = impurity
                best_split_point = split_point
                best_threshold = threshold
                best_shapelet = shapelet
            else:
                ts_view_free(&shapelet)

        # restore the best order to `samples`
        memcpy(
            self.samples + start,
            self.samples_buffer,
            sizeof(Py_ssize_t) * (end - start),
        )
        return new_split_point(best_split_point, best_threshold, best_shapelet)

    cdef Py_ssize_t _sample_shapelet(
        self,
        TSView *shapelet_info,
        Py_ssize_t start,
        Py_ssize_t end,
    ) nogil:
        cdef Py_ssize_t shapelet_length
        cdef Py_ssize_t shapelet_start
        cdef Py_ssize_t shapelet_index
        cdef Py_ssize_t shapelet_dim

        shapelet_length = rand_int(self.min_shapelet_size,
                                   self.max_shapelet_size,
                                   &self.random_seed)
        shapelet_start = rand_int(0, self.td.n_timestep - shapelet_length,
                                  &self.random_seed)
        shapelet_index = self.samples[rand_int(start, end, &self.random_seed)]
        if self.td.n_dims > 1:
            shapelet_dim = rand_int(0, self.td.n_dims, &self.random_seed)
        else:
            shapelet_dim = 1
        return self.distance_measure.init_ts_view(&self.td,
                                                  shapelet_info,
                                                  shapelet_index,
                                                  shapelet_start,
                                                  shapelet_length,
                                                  shapelet_dim)

    cdef void _partition_distance_buffer(self,
                                         Py_ssize_t start,
                                         Py_ssize_t end,
                                         Py_ssize_t *split_point,
                                         double *threshold,
                                         double *impurity) nogil:
        """ Partition the distance buffer such that an impurity measure is
        minimized
        
        Parameters
        ----------
        start : int
            The start position in samples
            
        end : int
            The end position in samples
            
        split_point : int (out) 
            The split point: an index in `range(start, end)` minimizing impurity

        threshold : float (out) 
            The threshold value

        impurity : float (in, out) 
            - The initial impurity (in) 
            - The optimal impurity (out)
            
        Warnings
        --------
        - Must be implemented by subclasses
        """
        with gil:
            raise NotImplementedError()


cdef class ClassificationShapeletTreeBuilder(ShapeletTreeBuilder):
    # the number of labels
    cdef Py_ssize_t n_labels

    # temporary buffer with sample distributions per label
    cdef double *label_buffer

    # temporary buffer with the left split sample distributions
    cdef double *left_label_buffer

    # temporary buffer with the right split sample distribution
    cdef double *right_label_buffer

    # the (strided) array of labels
    cdef Py_ssize_t *labels

    def __cinit__(
        self,
        Py_ssize_t n_shapelets,
        Py_ssize_t min_shapelet_size,
        Py_ssize_t max_shapelet_size,
        Py_ssize_t max_depth,
        Py_ssize_t min_sample_split,
        object metric,
        object metric_params,
        np.ndarray X,
        np.ndarray y,
        np.ndarray sample_weights,
        object random_state,
        Py_ssize_t n_labels
    ):
        self.labels = <Py_ssize_t*> y.data
        self.n_labels = n_labels
        self.tree = Tree(self.distance_measure, n_labels)
        self.label_buffer = <double*> malloc(sizeof(double) * n_labels)
        self.left_label_buffer = <double*> malloc(sizeof(double) * n_labels)
        self.right_label_buffer = <double*> malloc(sizeof(double) * n_labels)
        if (self.left_label_buffer == NULL or
                self.right_label_buffer == NULL or
                self.label_buffer == NULL):
            raise MemoryError()

    def __dealloc__(self):
        free(self.label_buffer)
        free(self.left_label_buffer)
        free(self.right_label_buffer)

    cdef Py_ssize_t new_leaf_node(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        Py_ssize_t parent,
        bint is_left,
    ) nogil:
        cdef Py_ssize_t node_id
        cdef Py_ssize_t i
        cdef double prob

        node_id = self.tree.add_leaf_node(
            parent, is_left, end - start, self.n_weighted_samples
        )
        for i in range(self.n_labels):
            prob = self.label_buffer[i] / self.n_weighted_samples
            self.tree.set_leaf_value(node_id, i, prob)
        return node_id

    cdef bint _is_pre_pruned(self, Py_ssize_t start, Py_ssize_t end) nogil:
        # reinitialize the `label_buffer` to the sample distribution
        # in the current sample region.
        memset(self.label_buffer, 0, sizeof(double) * self.n_labels)
        cdef Py_ssize_t n_positive = label_distribution(
            self.samples,
            self.sample_weights,
            start,
            end,
            self.labels,
            self.label_stride,
            self.n_labels,
            &self.n_weighted_samples,  # out param
            self.label_buffer,  # out param
        )

        if end - start <= self.min_sample_split or n_positive < 2:
            return True
        return False

    cdef void _partition_distance_buffer(
            self,
            Py_ssize_t start,
            Py_ssize_t end,
            Py_ssize_t *split_point,
            double *threshold,
            double *impurity,
    ) nogil:
        memset(self.left_label_buffer, 0, sizeof(double) * self.n_labels)

        # store the label buffer temporarily in `right_label_buffer`,
        # since all samples fall on the right hand side of the threshold
        memcpy(self.right_label_buffer, self.label_buffer,
               sizeof(double) * self.n_labels)

        cdef Py_ssize_t i  # real index of samples (in `range(start, end)`)
        cdef Py_ssize_t j  # sample index (in `samples`)
        cdef Py_ssize_t p  # label index (in `label_buffer`)

        cdef double right_sum
        cdef double left_sum

        cdef double prev_distance
        cdef Py_ssize_t prev_label

        cdef double current_sample_weight
        cdef double current_distance
        cdef double current_impurity
        cdef Py_ssize_t current_label

        j = self.samples[start]
        p = j * self.label_stride

        prev_distance = self.distance_buffer[start]
        prev_label = self.labels[p]

        if self.sample_weights != NULL:
            current_sample_weight = self.sample_weights[j]
        else:
            current_sample_weight = 1.0

        left_sum = current_sample_weight
        right_sum = self.n_weighted_samples - current_sample_weight

        self.left_label_buffer[prev_label] += current_sample_weight
        self.right_label_buffer[prev_label] -= current_sample_weight

        impurity[0] = entropy(left_sum,
                              self.left_label_buffer,
                              right_sum,
                              self.right_label_buffer,
                              self.n_labels)

        threshold[0] = prev_distance
        # The split point indicates a <=-relation
        split_point[0] = start + 1

        for i in range(start + 1, end - 1):
            j = self.samples[i]
            current_distance = self.distance_buffer[i]

            p = j * self.label_stride
            current_label = self.labels[p]

            if not current_label == prev_label:
                current_impurity = entropy(
                    left_sum,
                    self.left_label_buffer,
                    right_sum,
                    self.right_label_buffer,
                    self.n_labels,
                )

                if current_impurity <= impurity[0]:
                    impurity[0] = current_impurity
                    threshold[0] = (current_distance + prev_distance) / 2
                    split_point[0] = i

            if self.sample_weights != NULL:
                current_sample_weight = self.sample_weights[j]
            else:
                current_sample_weight = 1.0

            left_sum += current_sample_weight
            right_sum -= current_sample_weight
            self.left_label_buffer[current_label] += current_sample_weight
            self.right_label_buffer[current_label] -= current_sample_weight

            prev_label = current_label
            prev_distance = current_distance


cdef class RegressionShapeletTreeBuilder(ShapeletTreeBuilder):
    # the (strided) array of labels
    cdef double *labels
    cdef RollingVariance right
    cdef RollingVariance left

    def __cinit__(
        self,
        Py_ssize_t n_shapelets,
        Py_ssize_t min_shapelet_size,
        Py_ssize_t max_shapelet_size,
        Py_ssize_t max_depth,
        Py_ssize_t min_sample_split,
        object metric,
        object metric_params,
        np.ndarray X,
        np.ndarray y,
        np.ndarray sample_weights,
        object random_state,
    ):
        self.labels = <double*> y.data
        self.tree = Tree(self.distance_measure, 1)
        self.left = RollingVariance()
        self.right = RollingVariance()

    cdef bint _is_pre_pruned(self, Py_ssize_t start, Py_ssize_t end) nogil:
        cdef Py_ssize_t j
        cdef double sample_weight
        self.n_weighted_samples = 0
        for i in range(start, end):
            j = self.samples[i]
            if self.sample_weights != NULL:
                sample_weight = self.sample_weights[j]
            else:
                sample_weight = 1.0

            self.n_weighted_samples += sample_weight

        if end - start <= self.min_sample_split:
            return True
        return False

    cdef Py_ssize_t new_leaf_node(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        Py_ssize_t parent,
        bint is_left,
    ) nogil:
        cdef double leaf_sum = 0
        cdef double current_sample_weight = 0
        cdef Py_ssize_t j
        cdef Py_ssize_t node_id
        for i in range(start, end):
            j = self.samples[i]
            p = j * self.label_stride
            if self.sample_weights != NULL:
                current_sample_weight = self.sample_weights[j]
            else:
                current_sample_weight = 1.0
            leaf_sum += self.labels[p] * current_sample_weight

        node_id = self.tree.add_leaf_node(parent, is_left, end - start, self.n_weighted_samples)
        self.tree.set_leaf_value(node_id, 0, leaf_sum / self.n_weighted_samples)
        return node_id

    cdef void _partition_distance_buffer(self,
                                         Py_ssize_t start,
                                         Py_ssize_t end,
                                         Py_ssize_t *split_point,
                                         double *threshold,
                                         double *impurity) nogil:
        """Partitions the distance buffer into two binary partitions
        such that the sum of label variance in the two partitions is
        minimized.
       
        The implementation uses an efficient one-pass algorithm [1]
        for computing the variance of the two partitions and finding
        the optimal split point
        
        References
        ----------
        West, D. H. D. (1979). 
            Updating Mean and Variance Estimates: An Improved Method
        """
        cdef Py_ssize_t i  # real index of samples (in `range(start, end)`)
        cdef Py_ssize_t j  # sample index (in `samples`)
        cdef Py_ssize_t p  # label index (in `labels`)

        cdef double prev_distance

        cdef double current_sample_weight
        cdef double current_distance
        cdef double current_impurity
        cdef double current_val

        j = self.samples[start]
        p = j * self.label_stride

        prev_distance = self.distance_buffer[start]
        current_val = self.labels[p]

        if self.sample_weights != NULL:
            current_sample_weight = self.sample_weights[j]
        else:
            current_sample_weight = 1.0

        self.left._reset()
        self.right._reset()
        self.left._add(current_sample_weight, current_val)

        for i in range(start + 1, end):
            j = self.samples[i]
            p = j * self.label_stride
            if self.sample_weights != NULL:
                current_sample_weight = self.sample_weights[j]
            else:
                current_sample_weight = 1.0

            current_val = self.labels[p]
            self.right._add(current_sample_weight, current_val)

        impurity[0] = self.left._variance() + self.right._variance()
        threshold[0] = prev_distance
        split_point[0] = start + 1  # The split point indicates a <=-relation

        for i in range(start + 1, end - 1):
            j = self.samples[i]
            p = j * self.label_stride

            current_distance = self.distance_buffer[i]

            if self.sample_weights != NULL:
                current_sample_weight = self.sample_weights[j]
            else:
                current_sample_weight = 1.0

            # Move variance from the right-hand side to the left-hand
            # and reevaluate the impurity
            current_val = self.labels[p]
            self.right._remove(current_sample_weight, current_val)
            self.left._add(current_sample_weight, current_val)

            current_impurity = self.left._variance() + self.right._variance()
            if current_impurity <= impurity[0]:
                impurity[0] = current_impurity
                threshold[0] = (current_distance + prev_distance) / 2
                split_point[0] = i

            prev_distance = current_distance


cdef class ExtraRegressionShapeletTreeBuilder(RegressionShapeletTreeBuilder):

    cdef void _partition_distance_buffer(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        Py_ssize_t *split_point,
        double *threshold,
        double *impurity,
    ) nogil:
        # The smallest distance is always 0
        cdef double min_dist = self.distance_buffer[start + 1]
        cdef double max_dist = self.distance_buffer[end - 1]
        cdef double rand_threshold = rand_uniform(
            min_dist, max_dist, &self.random_seed
        )
        cdef Py_ssize_t i
        split_point[0] = start + 1
        for i in range(start + 1, end - 1):
            if self.distance_buffer[i] <= rand_threshold:
                split_point[0] = i
            else:
                break
        threshold[0] = rand_threshold
        # TODO: compute impurity scoring
        impurity[0] = 0


cdef class ExtraClassificationShapeletTreeBuilder(ClassificationShapeletTreeBuilder):

    cdef void _partition_distance_buffer(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        Py_ssize_t *split_point,
        double *threshold,
        double *impurity,
    ) nogil:
        # The smallest distance is always 0
        cdef double min_dist = self.distance_buffer[start + 1]
        cdef double max_dist = self.distance_buffer[end - 1]
        cdef double rand_threshold = rand_uniform(
            min_dist, max_dist, &self.random_seed
        )
        cdef Py_ssize_t i

        split_point[0] = start + 1
        for i in range(start + 1, end - 1):
            if self.distance_buffer[i] <= rand_threshold:
                split_point[0] = i
            else:
                break
        threshold[0] = rand_threshold
        # TODO: compute impurity scoring
        impurity[0] = -INFINITY
