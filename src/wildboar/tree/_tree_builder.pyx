# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3

# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wildboar is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
# General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# Authors: Isak Samsten

import numpy as np

cimport numpy as np
from libc.math cimport INFINITY, NAN, fabs, log2
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy, memset

from .._data cimport TSDatabase, ts_database_new
from .._utils cimport (
    RAND_R_MAX,
    RollingVariance,
    argsort,
    rand_int,
    rand_uniform,
    safe_realloc,
)
from ..embed._feature cimport Feature, FeatureEngineer


cpdef Tree _make_tree(
    FeatureEngineer feature_engineer,
    Py_ssize_t n_labels,
    list features,
    np.ndarray threshold,
    np.ndarray value,
    np.ndarray left,
    np.ndarray right,
    np.ndarray impurity,
    np.ndarray n_node_samples,
    np.ndarray n_weighted_node_samples
):
    cdef Tree tree = Tree(feature_engineer, n_labels, capacity=len(features) + 1)
    cdef Py_ssize_t node_count = len(features)
    cdef Py_ssize_t i
    cdef Py_ssize_t dim
    cdef np.ndarray arr
    cdef Feature *feature
    cdef np.ndarray value_reshape = value.reshape(-1)

    tree._node_count = node_count
    for i in range(node_count):
        if features[i] is not None:
            feature = <Feature*> malloc(sizeof(Feature))
            feature_engineer.persistent_feature_from_object(features[i], feature)
            tree._features[i] = feature
        else:
            tree._features[i] = NULL
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
        FeatureEngineer feature_engineer,
        Py_ssize_t n_labels,
        Py_ssize_t capacity=10
    ):
        self.feature_engineer = feature_engineer
        self._node_count = 0
        self._capacity = capacity
        self._n_labels = n_labels
        self._features = <Feature**> malloc(self._capacity * sizeof(Feature*))
        self._thresholds = <double*> malloc(self._capacity * sizeof(double))
        self._values = <double*> malloc(self._capacity * self._n_labels * sizeof(double))
        self._left = <Py_ssize_t*> malloc(self._capacity * sizeof(Py_ssize_t))
        self._right = <Py_ssize_t*> malloc(self._capacity * sizeof(Py_ssize_t))
        self._impurity = <double*> malloc(self._capacity * sizeof(double))
        self._n_node_samples = <Py_ssize_t*> malloc(self._capacity * sizeof(Py_ssize_t))
        self._n_weighted_node_samples = <double*> malloc(self._capacity * sizeof(double))

    def __dealloc__(self):
        cdef Py_ssize_t i
        if self._features != NULL:
            for i in range(self._node_count):
                if self._features[i] != NULL:
                    self.feature_engineer.free_persistent_feature(self._features[i])
                    free(self._features[i])
            free(self._features)

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
            self.feature_engineer,
            self._n_labels,
            self.features,
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
    def features(self):
        cdef Py_ssize_t i, j
        cdef Feature* feature
        cdef object object
        cdef list ret = []
        for i in range(self._node_count):
            feature = self._features[i]
            if feature != NULL:
                object = self.feature_engineer.persistent_feature_to_object(feature)
                ret.append(object)
            else:
                ret.append(None)
        return ret

    @property
    def n_node_samples(self):
        cdef np.ndarray arr = np.zeros(self._node_count, dtype=int)
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
        cdef np.ndarray arr = np.empty(self._node_count, dtype=int)
        cdef Py_ssize_t i
        for i in range(self._node_count):
            arr[i] = self._left[i]
        return arr

    @property
    def right(self):
        cdef np.ndarray arr = np.empty(self._node_count, dtype=int)
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
        cdef long *out_data = <long*> out.data
        cdef Feature *feature
        cdef double threshold, feature_value
        cdef int node_index
        cdef Py_ssize_t i
        with nogil:
            for i in range(ts.n_samples):
                node_index = 0
                while self._left[node_index] != -1:
                    threshold = self._thresholds[node_index]
                    feature = self._features[node_index]
                    feature_value = self.feature_engineer.persistent_feature_value(
                        feature, &ts, i
                    )
                    if feature_value <= threshold:
                        node_index = self._left[node_index]
                    else:
                        node_index = self._right[node_index]
                out_data[i] = <long> node_index
        return out

    cpdef np.ndarray decision_path(self, object X):
        if not isinstance(X, np.ndarray):
            raise ValueError(f"X should be np.ndarray, got {type(X)}")
        cdef TSDatabase ts = ts_database_new(X)
        cdef np.ndarray out = np.zeros((ts.n_samples, self.node_count), order="c", dtype=np.intp)

        cdef long *out_data = <long*> out.data
        cdef Py_ssize_t i_stride = <Py_ssize_t> out.strides[0] / <Py_ssize_t> out.itemsize
        cdef Py_ssize_t n_stride = <Py_ssize_t> out.strides[1] / <Py_ssize_t> out.itemsize
        cdef Py_ssize_t node_index
        cdef Py_ssize_t i
        cdef Feature *feature
        cdef double threshold, feature_value
        with nogil:
            for i in range(ts.n_samples):
                node_index = 0
                while self._left[node_index] != -1:
                    out_data[i * i_stride + node_index * n_stride] = 1
                    threshold = self._thresholds[node_index]
                    feature = self._features[node_index]
                    feature_value = self.feature_engineer.persistent_feature_value(
                        feature, &ts, i
                    )
                    if feature_value <= threshold:
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
        self._features[node_id] = NULL
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
        Feature *feature,
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
        self._features[node_id] = feature
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
        ret = safe_realloc(<void**> &self._features, sizeof(Feature*) * new_capacity)
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
    Feature feature,
) nogil:
    cdef SplitPoint s
    s.split_point = split_point
    s.threshold = threshold
    s.feature = feature
    return s


cdef inline Py_ssize_t label_distribution(
    const Py_ssize_t *samples,
    const double *sample_weights,
    Py_ssize_t start,
    Py_ssize_t end,
    const Py_ssize_t *labels,
    Py_ssize_t label_stride,
    Py_ssize_t n_labels,
    double *n_weighted_samples,
    double *label_dist,
) nogil:
    cdef double sample_weight
    cdef Py_ssize_t i, j, p, n_pos

    n_pos = 0
    n_weighted_samples[0] = 0
    for i in range(start, end):
        j = samples[i]
        p = j * label_stride

        if sample_weights != NULL:
            sample_weight = sample_weights[j]
        else:
            sample_weight = 1.0

        label_dist[labels[p]] += sample_weight
        n_weighted_samples[0] += sample_weight

    for i in range(n_labels):
        if label_dist[i] > 0:
            n_pos += 1

    return n_pos


cdef inline double entropy(
    double  left_sum,
    double *left_count,
    double  right_sum,
    double *right_count,
    Py_ssize_t  n_labels
) nogil:
    cdef double n_samples = left_sum + right_sum
    cdef double x_sum = 0
    cdef double y_sum = 0
    cdef double xv, yv
    cdef Py_ssize_t i
    for i in range(n_labels):
        xv = left_count[i] / n_samples
        yv = right_count[i] / n_samples
        if xv > 0:
            x_sum += xv * log2(xv)
        if yv > 0:
            y_sum += yv * log2(yv)

    return fabs(
        (left_sum / n_samples) * -x_sum +
        (right_sum / n_samples) * -y_sum
    )


cdef class TreeBuilder:

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

    # temporary buffer for feature computations
    cdef double *feature_buffer

    # the feature measure implementation
    cdef FeatureEngineer feature_engineer

    # the tree structure representation
    cdef Tree tree

    cdef size_t random_seed

    def __cinit__(
        self,
        Py_ssize_t max_depth,
        Py_ssize_t min_sample_split,
        np.ndarray X,
        np.ndarray y,
        np.ndarray sample_weights,
        FeatureEngineer feature_engineer,
        object random_state,
        *args,
        **kwargs
    ):
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.current_node_id = 0
        self.random_seed = random_state.randint(0, RAND_R_MAX)

        self.td = ts_database_new(X)
        self.feature_engineer = feature_engineer
        self.label_stride = <Py_ssize_t> y.strides[0] / <Py_ssize_t> y.itemsize

        self.n_samples = X.shape[0]
        self.samples = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * self.n_samples)
        self.samples_buffer = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * self.n_samples)
        self.feature_buffer = <double*> malloc(sizeof(double) * self.n_samples)

        if (
            self.samples == NULL or
            self.feature_buffer == NULL or
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
        free(self.feature_buffer)

    @property
    def tree_(self):
        return self.tree

    cpdef int build_tree(self):
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
        int parent,
        bint is_left,
    ) nogil:
        pass

    cdef Py_ssize_t new_branch_node(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        SplitPoint sp,
        Feature *persistent_feature,
        int parent,
        bint is_left,
    ) nogil:
        cdef Py_ssize_t node_id
        node_id = self.tree.add_branch_node(
            parent,
            is_left,
            end - start,
            self.n_weighted_samples,
            persistent_feature,
            sp.threshold, 
            -1,
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
        int parent,
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
        
        cdef Feature *persistent_feature
        cdef Py_ssize_t current_node_id, left_node_id, right_node_id
        cdef Py_ssize_t err
        if split.split_point > start and end - split.split_point > 0:
            # The persistent feature is freed by the Tree
            persistent_feature = <Feature*> malloc(sizeof(Feature))
            err = self.feature_engineer.init_persistent_feature(
                &self.td, &split.feature, persistent_feature)
            self.feature_engineer.free_transient_feature(&split.feature)
            if err == -1:
                return -1

            current_node_id = self.new_branch_node(
                start, end, split, persistent_feature, parent, is_left
            )
            left_node_id = self._build_tree(
                start, split.split_point, depth + 1, current_node_id, True, max_depth
            )
            right_node_id = self._build_tree(
                split.split_point, end, depth + 1, current_node_id, False, max_depth
            )
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
         - `self._partition_feature_buffer`: implements the decision 
            to determine the split quality minimizing `impurity`
        """
        cdef Py_ssize_t split_point, best_split_point
        cdef double threshold, best_threshold
        cdef double impurity
        cdef double best_impurity
        cdef Feature feature
        cdef Feature best_feature
        cdef Py_ssize_t i, n_samples

        feature.feature = NULL
        best_feature.feature = NULL

        n_samples = end - start
        best_impurity = INFINITY
        best_threshold = NAN
        best_split_point = 0
        split_point = 0
        for i in range(self.feature_engineer.get_n_features(&self.td)):
            self.feature_engineer.next_feature(
                i, &self.td, self.samples + start, n_samples, &feature, &self.random_seed)
            
            self.feature_engineer.transient_feature_values(
                &feature,
                &self.td,
                self.samples + start,
                end - start,
                self.feature_buffer + start,
            )
            argsort(self.feature_buffer + start, self.samples + start, n_samples)
            self._partition_feature_buffer(
                start, end, &split_point, &threshold, &impurity)
            if impurity < best_impurity:
                # store the order of samples in `sample_buffer`
                memcpy(
                    self.samples_buffer,
                    self.samples + start,
                    sizeof(Py_ssize_t) * n_samples,
                )
                best_impurity = impurity
                best_split_point = split_point
                best_threshold = threshold
                self.feature_engineer.free_transient_feature(&best_feature)
                best_feature = feature
            else:
                self.feature_engineer.free_transient_feature(&feature)

        # restore the best order to `samples`
        memcpy(
            self.samples + start,
            self.samples_buffer,
            sizeof(Py_ssize_t) * n_samples,
        )
        return new_split_point(best_split_point, best_threshold, best_feature)

    cdef void _partition_feature_buffer(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        Py_ssize_t *split_point,
        double *threshold,
        double *impurity,
    ) nogil:
        """ Partition the feature buffer such that an impurity measure is
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


cdef class ClassificationTreeBuilder(TreeBuilder):
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
        Py_ssize_t max_depth,
        Py_ssize_t min_sample_split,
        np.ndarray X,
        np.ndarray y,
        np.ndarray sample_weights,
        FeatureEngineer feature_engineer,
        object random_state,
        Py_ssize_t n_labels,
        *args,
        **kwargs,
    ):
        self.labels = <Py_ssize_t*> y.data
        self.n_labels = n_labels
        self.tree = Tree(self.feature_engineer, n_labels) # TODO
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
        int parent,
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

    cdef void _partition_feature_buffer(
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

        cdef double prev_feature
        cdef Py_ssize_t prev_label

        cdef double current_sample_weight
        cdef double current_feature
        cdef double current_impurity
        cdef Py_ssize_t current_label

        j = self.samples[start]
        p = j * self.label_stride

        prev_feature = self.feature_buffer[start]
        prev_label = self.labels[p]

        if self.sample_weights != NULL:
            current_sample_weight = self.sample_weights[j]
        else:
            current_sample_weight = 1.0

        left_sum = current_sample_weight
        right_sum = self.n_weighted_samples - current_sample_weight

        self.left_label_buffer[prev_label] += current_sample_weight
        self.right_label_buffer[prev_label] -= current_sample_weight

        impurity[0] = entropy(
            left_sum,
            self.left_label_buffer,
            right_sum,
            self.right_label_buffer,
            self.n_labels
        )

        threshold[0] = prev_feature
        # The split point indicates a <=-relation
        split_point[0] = start + 1

        for i in range(start + 1, end - 1):
            j = self.samples[i]
            current_feature = self.feature_buffer[i]

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
                    threshold[0] = (current_feature + prev_feature) / 2
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
            prev_feature = current_feature


cdef class RegressionTreeBuilder(TreeBuilder):
    # the (strided) array of labels
    cdef double *labels
    cdef RollingVariance right
    cdef RollingVariance left

    def __cinit__(
        self,
        Py_ssize_t max_depth,
        Py_ssize_t min_sample_split,
        np.ndarray X,
        np.ndarray y,
        np.ndarray sample_weights,
        FeatureEngineer feature_engineer,
        *args,
        **kwargs,
    ):
        self.labels = <double*> y.data
        self.tree = Tree(self.feature_engineer, 1) # TODO
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
        int parent,
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

    cdef void _partition_feature_buffer(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        Py_ssize_t *split_point,
        double *threshold,
        double *impurity,
    ) nogil:
        """Partitions the feature buffer into two binary partitions
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

        cdef double prev_feature

        cdef double current_sample_weight
        cdef double current_feature
        cdef double current_impurity
        cdef double current_val

        j = self.samples[start]
        p = j * self.label_stride

        prev_feature = self.feature_buffer[start]
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
        threshold[0] = prev_feature
        split_point[0] = start + 1  # The split point indicates a <=-relation

        for i in range(start + 1, end - 1):
            j = self.samples[i]
            p = j * self.label_stride

            current_feature = self.feature_buffer[i]

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
                threshold[0] = (current_feature + prev_feature) / 2
                split_point[0] = i

            prev_feature = current_feature


cdef class ExtraRegressionTreeBuilder(RegressionTreeBuilder):

    cdef void _partition_feature_buffer(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        Py_ssize_t *split_point,
        double *threshold,
        double *impurity,
    ) nogil:
        # The smallest feature is always 0
        cdef double min_feature = self.feature_buffer[start + 1]
        cdef double max_feature = self.feature_buffer[end - 1]
        cdef double rand_threshold = rand_uniform(
            min_feature, max_feature, &self.random_seed
        )
        cdef Py_ssize_t i
        split_point[0] = start + 1
        for i in range(start + 1, end - 1):
            if self.feature_buffer[i] <= rand_threshold:
                split_point[0] = i
            else:
                break
        threshold[0] = rand_threshold
        # TODO: compute impurity scoring
        impurity[0] = 0


cdef class ExtraClassificationTreeBuilder(ClassificationTreeBuilder):

    cdef void _partition_feature_buffer(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        Py_ssize_t *split_point,
        double *threshold,
        double *impurity,
    ) nogil:
        # TODO: is this still true?
        # The smallest feature is always 0
        cdef double min_feature = self.feature_buffer[start + 1]
        cdef double max_feature = self.feature_buffer[end - 1]
        cdef double rand_threshold = rand_uniform(
            min_feature, max_feature, &self.random_seed
        )
        cdef Py_ssize_t i

        split_point[0] = start + 1
        for i in range(start + 1, end - 1):
            if self.feature_buffer[i] <= rand_threshold:
                split_point[0] = i
            else:
                break
        threshold[0] = rand_threshold
        # TODO: compute impurity scoring
        impurity[0] = -INFINITY