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
from libc.math cimport INFINITY, NAN, log2
from libc.stdlib cimport calloc, free, malloc
from libc.string cimport memcpy, memset

from scipy.sparse import csr_matrix

from wildboar.distance._distance cimport DistanceMeasure

from wildboar.distance import _DISTANCE_MEASURE

from wildboar.embed._feature cimport Feature, FeatureEngineer

from wildboar.utils import check_dataset

from wildboar.utils.data cimport Dataset
from wildboar.utils.misc cimport CList, argsort, safe_realloc
from wildboar.utils.rand cimport RAND_R_MAX, rand_int, rand_uniform


cdef double FEATURE_THRESHOLD = 1e-7

cdef class Criterion:

    cdef double weighted_n_left
    cdef double weighted_n_right
    cdef double weighted_n_total
    cdef Py_ssize_t start
    cdef Py_ssize_t end
    cdef Py_ssize_t *samples
    cdef double *sample_weight

    cdef void init(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        Py_ssize_t *samples,
        double *sample_weights,
    ) nogil:
        self.start = start
        self.end = end
        self.samples = samples
        self.sample_weight = sample_weights

    cdef void reset(self) nogil:
        pass

    cdef void update(self, Py_ssize_t pos, Py_ssize_t new_pos) nogil:
        pass

    cdef double proxy_impurity(self) nogil:
        cdef double left_impurity
        cdef double right_impurity
        self.child_impurity(&left_impurity, &right_impurity)
        return -self.weighted_n_right * right_impurity - self.weighted_n_left * left_impurity

    cdef double impurity(self) nogil:
        pass

    cdef void child_impurity(self, double *left, double *right) nogil:
        pass

    cdef double impurity_improvement(
        self,
        double impurity_parent,
        double impurity_left,
        double impurity_right,
        double weighted_n_samples,
    ) nogil:
        return ((self.weighted_n_total / weighted_n_samples) *
                (impurity_parent - (self.weighted_n_right / self.weighted_n_total * impurity_right)
                                 - (self.weighted_n_left / self.weighted_n_total * impurity_left)))

    cdef void leaf_value(self, Tree tree, Py_ssize_t node_id) nogil:
        pass

cdef class ClassificationCriterion(Criterion):

    cdef Py_ssize_t *labels
    cdef Py_ssize_t label_stride
    cdef Py_ssize_t n_labels
    cdef double *sum_left
    cdef double *sum_right
    cdef double *sum_total

    def __cinit__(self, np.ndarray y, Py_ssize_t n_labels):
        if y.dtype != np.intp:
            raise ValueError("unexpected dtype (%r != %r)" % (y.dtype, np.intp))

        if y.ndim != 1:
            raise ValueError("unexpected dim (%r != 1)" % y.ndim)
        self.labels = <Py_ssize_t*> y.data
        self.label_stride = <Py_ssize_t> y.strides[0] / <Py_ssize_t> y.itemsize
        self.n_labels = n_labels
        self.sum_left = <double*> calloc(n_labels, sizeof(double))
        self.sum_right = <double*> calloc(n_labels, sizeof(double))
        self.sum_total = <double*> calloc(n_labels, sizeof(double))

    def __dealloc__(self):
        free(self.sum_total)
        free(self.sum_left)
        free(self.sum_right)

    cdef void init(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        Py_ssize_t *samples,
        double *sample_weights,
    ) nogil:
        Criterion.init(self, start, end, samples, sample_weights)
        self.weighted_n_total = 0

        memset(self.sum_total, 0, self.n_labels * sizeof(double))

        cdef Py_ssize_t i, j, p
        cdef double w = 1.0
        for i in range(start, end):
            j = samples[i]
            p = j * self.label_stride

            if sample_weights != NULL:
                w = sample_weights[j]

            self.sum_total[self.labels[p]] += w
            self.weighted_n_total += w

        self.reset()

    cdef void reset(self) nogil:
        self.weighted_n_left = 0
        self.weighted_n_right = self.weighted_n_total
        memset(self.sum_left, 0, self.n_labels * sizeof(double))
        memcpy(self.sum_right, self.sum_total, self.n_labels * sizeof(double))

    cdef void update(self, Py_ssize_t pos, Py_ssize_t new_pos) nogil:
        cdef Py_ssize_t i, j, p
        cdef double w = 1.0

        for i in range(pos, new_pos):
            j = self.samples[i]
            p = j * self.label_stride

            if self.sample_weight != NULL:
                w = self.sample_weight[j]

            self.sum_left[self.labels[p]] += w
            self.weighted_n_left += w

        self.weighted_n_right = self.weighted_n_total - self.weighted_n_left
        for i in range(self.n_labels):
            self.sum_right[i] = self.sum_total[i] - self.sum_left[i]

    cdef double impurity(self) nogil:
        pass

    cdef void child_impurity(self, double* left, double *right) nogil:
        pass

    cdef void leaf_value(self, Tree tree, Py_ssize_t node_id) nogil:
        cdef Py_ssize_t i
        cdef double prob
        for i in range(self.n_labels):
            prob = self.sum_total[i] / self.weighted_n_total
            tree.set_leaf_value(node_id, i, prob)

cdef class GiniCriterion(ClassificationCriterion):

    cdef double impurity(self) nogil:
        cdef double gini = 0.0
        cdef double c
        cdef Py_ssize_t i
        for i in range(self.n_labels):
            c = self.sum_total[i]
            gini += 1.0 - (c * c) / (self.weighted_n_total * self.weighted_n_total)

        return gini

    cdef void child_impurity(self, double *left, double *right) nogil:
        cdef double sq_left = 0
        cdef double sq_right = 0
        cdef double v
        cdef Py_ssize_t i

        for i in range(self.n_labels):
            v = self.sum_left[i]
            sq_left += v * v

            v = self.sum_right[i]
            sq_right += v * v

        left[0] = 1 - sq_left / (self.weighted_n_left * self.weighted_n_left)
        right[0] = 1 - sq_right / (self.weighted_n_right * self.weighted_n_right)

cdef class EntropyCriterion(ClassificationCriterion):

    cdef double impurity(self) nogil:
        cdef double c
        cdef double entropy = 0
        cdef Py_ssize_t i
        for i in range(self.n_labels):
            c = self.sum_total[i]
            if c > 0:
                c /= self.weighted_n_total
                entropy -= c * log2(c)

        return entropy

    cdef void child_impurity(self, double *left, double *right) nogil:
        left[0] = 0
        right[0] = 0
        cdef double v
        cdef Py_ssize_t i

        for i in range(self.n_labels):
            v = self.sum_left[i]
            if v > 0:
                v /= self.weighted_n_left
                left[0] -= v * log2(v)

            v = self.sum_right[i]
            if v > 0:
                v /= self.weighted_n_right
                right[0] -= v * log2(v)

cdef class RegressionCriterion(Criterion):
    cdef double sum_left
    cdef double sum_right
    cdef double sum_total
    cdef double sum_sq_total
    cdef double *labels
    cdef Py_ssize_t label_stride
    cdef Py_ssize_t pos

    def __cinit__(self, np.ndarray y):
        if y.dtype != np.double:
            raise ValueError("unexpected dtype (%r != np.double)" % y.dtype)

        if y.ndim != 1:
            raise ValueError("unexpected dim (%r != 1)" % y.ndim)        

        self.label_stride = <Py_ssize_t> y.strides[0] / <Py_ssize_t> y.itemsize
        self.labels = <double*> y.data

        self.sum_left = 0
        self.sum_right = 0
        self.sum_total = 0
        self.sum_sq_total = 0

    cdef void init(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        Py_ssize_t *samples,
        double *sample_weights,
    ) nogil:
        Criterion.init(self, start, end, samples, sample_weights)
        self.sum_total = 0
        self.sum_sq_total = 0
        self.weighted_n_total = 0

        cdef Py_ssize_t i, j, p
        cdef double x
        cdef double w = 1.0

        for i in range(start, end):
            j = samples[i]
            p = j * self.label_stride
            if sample_weights != NULL:
                w = sample_weights[j]

            x = w * self.labels[p]
            self.sum_total += x
            self.sum_sq_total += x * x
            self.weighted_n_total += w

        self.reset()
        self.start = start

    cdef void reset(self) nogil:
        self.weighted_n_left = 0
        self.weighted_n_right = self.weighted_n_total
        self.sum_left = 0
        self.sum_right = self.sum_total
        self.pos = 0

    cdef void update(self, Py_ssize_t pos, Py_ssize_t new_pos) nogil:
        cdef Py_ssize_t i
        cdef Py_ssize_t j
        cdef Py_ssize_t p
        cdef double w = 1.0
        for i in range(pos, new_pos):
            j = self.samples[i]
            p = j * self.label_stride

            if self.sample_weight != NULL:
                w = self.sample_weight[j]

            self.sum_left += w * self.labels[p]
            self.weighted_n_left += w

        self.weighted_n_right = self.weighted_n_total - self.weighted_n_left
        self.sum_right = self.sum_total - self.sum_left
        self.pos = new_pos

    cdef void leaf_value(self, Tree tree, Py_ssize_t node_id) nogil:
        tree.set_leaf_value(node_id, 0, self.sum_total / self.weighted_n_total)

cdef class MSECriterion(RegressionCriterion):

    cdef double proxy_impurity(self) nogil:
        cdef double proxy_impurity_left = self.sum_left * self.sum_left
        cdef double proxy_impurity_right = self.sum_right * self.sum_right
        return proxy_impurity_left / self.weighted_n_left + proxy_impurity_right / self.weighted_n_right

    cdef double impurity(self) nogil:
        cdef double impurity
        impurity = self.sum_sq_total / self.weighted_n_total
        impurity -= (self.sum_total / self.weighted_n_total) ** 2
        return impurity

    cdef void child_impurity(self, double* left, double *right) nogil:
        cdef double left_sq_sum = 0
        cdef double right_sq_sum = 0
        cdef double w = 1.0
        cdef double y
        cdef Py_ssize_t i, j, p

        for i in range(self.start, self.pos):
            j = self.samples[i]
            p = self.label_stride * j

            if self.sample_weight != NULL:
                w = self.sample_weight[j]

            y = self.labels[p]
            left_sq_sum += w * y * y
        right_sq_sum = self.sum_sq_total - left_sq_sum

        left[0] = left_sq_sum / self.weighted_n_left
        left[0] -= (self.sum_left / self.weighted_n_left) ** 2
        right[0] = right_sq_sum / self.weighted_n_right
        right[0] -= (self.sum_right / self.weighted_n_right) ** 2

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
        cdef np.ndarray arr = np.empty(self._node_count * self._n_labels, dtype=float)
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
        cdef np.ndarray arr = np.zeros(self._node_count, dtype=np.intp)
        cdef Py_ssize_t i
        for i in range(self._node_count):
            arr[i] = self._n_node_samples[i]
        return arr

    @property
    def n_weighted_node_samples(self):
        cdef np.ndarray arr = np.zeros(self._node_count, dtype=float)
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
        cdef np.ndarray arr = np.empty(self._node_count, dtype=float)
        cdef Py_ssize_t i
        for i in range(self._node_count):
            arr[i] = self._thresholds[i]
        return arr

    @property
    def impurity(self):
        cdef np.ndarray arr = np.empty(self._node_count, dtype=float)
        cdef Py_ssize_t i
        for i in range(self._node_count):
            arr[i] = self._impurity[i]
        return arr

    def predict(self, object X):
        cdef np.ndarray apply = self.apply(X)
        cdef np.ndarray predict = np.take(self.value, apply, axis=0, mode="clip")
        if self._n_labels == 1:
            predict = predict.reshape(X.shape[0])
        return predict

    def apply(self, object X):
        if not isinstance(X, np.ndarray):
            raise ValueError(f"X should be np.ndarray, got {type(X)}")
        X = check_dataset(X)
        cdef Dataset ts = Dataset(X)
        cdef np.ndarray out = np.zeros((ts.n_samples,), dtype=np.intp)
        cdef Py_ssize_t *out_data = <Py_ssize_t*> out.data
        cdef Feature *feature
        cdef double threshold, feature_value
        cdef Py_ssize_t node_index
        cdef Py_ssize_t i
        with nogil:
            self.feature_engineer.reset(ts)
            for i in range(ts.n_samples):
                node_index = 0
                while self._left[node_index] != -1:
                    threshold = self._thresholds[node_index]
                    feature = self._features[node_index]
                    feature_value = self.feature_engineer.persistent_feature_value(
                        feature, ts, i
                    )
                    if feature_value <= threshold:
                        node_index = self._left[node_index]
                    else:
                        node_index = self._right[node_index]
                out_data[i] = <Py_ssize_t> node_index
        return out

    def decision_path(self, object X):
        if not isinstance(X, np.ndarray):
            raise ValueError(f"X should be np.ndarray, got {type(X)}")
        X = check_dataset(X)
        cdef Dataset ts = Dataset(X)
        cdef np.ndarray out = np.zeros((ts.n_samples, self.node_count), order="c", dtype=np.intc)

        cdef int *out_data = <int*> out.data
        cdef Py_ssize_t i_stride = <Py_ssize_t> out.strides[0] / <Py_ssize_t> out.itemsize
        cdef Py_ssize_t n_stride = <Py_ssize_t> out.strides[1] / <Py_ssize_t> out.itemsize
        cdef Py_ssize_t node_index
        cdef Py_ssize_t i
        cdef Feature *feature
        cdef double threshold, feature_value
        with nogil:
            self.feature_engineer.reset(ts)
            for i in range(ts.n_samples):
                node_index = 0
                while self._left[node_index] != -1:
                    out_data[i * i_stride + node_index * n_stride] = 1
                    threshold = self._thresholds[node_index]
                    feature = self._features[node_index]
                    feature_value = self.feature_engineer.persistent_feature_value(
                        feature, ts, i
                    )
                    if feature_value <= threshold:
                        node_index = self._left[node_index]
                    else:
                        node_index = self._right[node_index]
        return csr_matrix(out, dtype=bool)

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
        self._capacity = new_capacity
        return 0

cdef class TreeBuilder:

    # hyper-parameters
    cdef Py_ssize_t max_depth
    cdef Py_ssize_t min_sample_split
    cdef Py_ssize_t min_sample_leaf
    cdef double min_impurity_decrease

    # the id (in Tree) of the current node
    cdef Py_ssize_t current_node_id

    # the stride of the label array
    cdef Py_ssize_t label_stride

    # the weight of the j:th sample
    cdef double *sample_weights

    # the dataset of time series
    cdef Dataset td

    # the number of samples with non-zero weight
    cdef Py_ssize_t n_samples
    cdef double n_weighted_samples

    # buffer of samples from 0, ..., n_samples
    cdef Py_ssize_t *samples

    # temporary buffer of samples from 0, ..., n_samples
    cdef Py_ssize_t *samples_buffer

    # temporary buffer for feature computations
    cdef double *feature_buffer

    cdef FeatureEngineer feature_engineer
    cdef Criterion criterion
    cdef Tree tree
    cdef size_t random_seed

    def __cinit__(
        self,
        np.ndarray X,
        np.ndarray sample_weights,
        FeatureEngineer feature_engineer,
        Criterion criterion,
        Tree tree,
        object random_state,
        Py_ssize_t max_depth=2**16,
        Py_ssize_t min_sample_split=2,
        Py_ssize_t min_sample_leaf=1,
        double min_impurity_decrease=0.0,
    ):
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.min_sample_leaf = min_sample_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_seed = random_state.randint(0, RAND_R_MAX)

        self.td = Dataset(X)
        self.feature_engineer = feature_engineer
        self.criterion = criterion
        self.tree = tree

        self.current_node_id = 0
        self.n_samples = self.td.n_samples
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
        self.n_weighted_samples = 0.0
        for i in range(self.n_samples):
            if sample_weights is None or sample_weights[i] != 0.0:
                self.samples[j] = i
                j += 1
                if sample_weights is not None:
                    self.n_weighted_samples += sample_weights[i]
                else:
                    self.n_weighted_samples += 1.0

        self.n_samples = j

        if sample_weights is None:
            self.sample_weights = NULL
        else:
            if sample_weights.dtype != np.double:
                raise ValueError("unexpected dtype (%r != np.double)" % sample_weights.dtype)
            if sample_weights.ndim != 1:
                raise ValueError("unexpected dim (%r != 1)" % sample_weights.ndim)
            if sample_weights.strides[0] // sample_weights.itemsize != 1:
                raise ValueError("unexpected stride")

            self.sample_weights = <double*> sample_weights.data

    def __dealloc__(self):
        free(self.samples)
        free(self.samples_buffer)
        free(self.feature_buffer)

    @property
    def tree_(self):
        return self.tree

    cpdef Py_ssize_t build_tree(self):
        cdef Py_ssize_t root_node_id
        cdef Py_ssize_t max_depth = 0
        with nogil:
            self.feature_engineer.reset(self.td)
            root_node_id = self._build_tree(
                0,
                self.n_samples,
                0,
                -1,
                False,
                NAN,
                &max_depth
            )
        self.tree._max_depth = max_depth
        return root_node_id

    cdef Py_ssize_t new_leaf_node(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        Py_ssize_t parent,
        bint is_left,
    ) nogil:
        cdef Py_ssize_t node_id = self.tree.add_leaf_node(
            parent, is_left, end - start, self.criterion.weighted_n_total
        )
        self.criterion.leaf_value(self.tree, node_id)
        return node_id

    cdef Py_ssize_t new_branch_node(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        SplitPoint sp,
        Feature *persistent_feature,
        Py_ssize_t parent,
        bint is_left,
    ) nogil:
        cdef Py_ssize_t node_id
        node_id = self.tree.add_branch_node(
            parent,
            is_left,
            end - start,
            self.criterion.weighted_n_total,
            persistent_feature,
            sp.threshold,
            sp.impurity_improvement,
        )
        return node_id

    cdef Py_ssize_t _build_tree(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        Py_ssize_t depth,
        Py_ssize_t parent,
        bint is_left,
        double impurity,
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

        self.criterion.init(start, end, self.samples, self.sample_weights)
        cdef Py_ssize_t n_node_samples = end - start
        cdef bint is_leaf = (
            depth >= self.max_depth
            or n_node_samples < self.min_sample_split
            or n_node_samples < 2 * self.min_sample_leaf
        )
        if is_leaf:
            return self.new_leaf_node(start, end, parent, is_left)

        if parent < 0:
            impurity = self.criterion.impurity()

        cdef SplitPoint split = self._split(start, end, impurity)

        cdef Feature *persistent_feature
        cdef Py_ssize_t current_node_id
        cdef Py_ssize_t err
        is_leaf = (
            split.split_point <= start
            or split.split_point >= end
            or split.impurity_improvement <= self.min_impurity_decrease
        )
        if not is_leaf:
            # The persistent feature is freed by the Tree
            persistent_feature = <Feature*> malloc(sizeof(Feature))
            err = self.feature_engineer.init_persistent_feature(
                self.td, &split.feature, persistent_feature
            )
            self.feature_engineer.free_transient_feature(&split.feature)
            if err == -1:
                return -1

            current_node_id = self.new_branch_node(
                start, end, split, persistent_feature, parent, is_left
            )
            self._build_tree(
                start,
                split.split_point,
                depth + 1,
                current_node_id,
                True,
                split.impurity_left,
                max_depth,
            )
            self._build_tree(
                split.split_point,
                end,
                depth + 1,
                current_node_id,
                False,
                split.impurity_right,
                max_depth,
            )
            return current_node_id
        else:
            return self.new_leaf_node(start, end, parent, is_left)

    cdef SplitPoint _split(self, Py_ssize_t start, Py_ssize_t end, double parent_impurity) nogil:
        cdef Py_ssize_t i, n_samples
        cdef Py_ssize_t current_split_point
        cdef double current_threshold
        cdef double current_impurity
        cdef double best_impurity
        cdef Feature current_feature

        n_samples = end - start

        best_impurity = -INFINITY

        current_feature.feature = NULL
        current_impurity = -INFINITY
        current_threshold = NAN
        current_split_point = 0

        cdef SplitPoint best
        best.threshold = NAN
        best.split_point = 0
        best.feature.feature = NULL

        for i in range(self.feature_engineer.get_n_features(self.td)):
            self.feature_engineer.next_feature(
                i, self.td, self.samples + start, n_samples, &current_feature, &self.random_seed)
            
            self.feature_engineer.transient_feature_values(
                &current_feature,
                self.td,
                self.samples + start,
                end - start,
                self.feature_buffer + start,
            )
            argsort(self.feature_buffer + start, self.samples + start, n_samples)

            # All feature values are constant
            if self.feature_buffer[end - 1] <= self.feature_buffer[start] + FEATURE_THRESHOLD:
                continue

            self._partition_feature_buffer(
                start, end, &current_split_point, &current_threshold, &current_impurity
            )
            if current_impurity > best_impurity:
                # store the order of samples in `sample_buffer`
                memcpy(
                    self.samples_buffer,
                    self.samples + start,
                    sizeof(Py_ssize_t) * n_samples,
                )
                best_impurity = current_impurity

                best.split_point = current_split_point
                best.threshold = current_threshold
                if best.feature.feature != NULL:
                    self.feature_engineer.free_transient_feature(&best.feature)
                best.feature = current_feature
            else:
                self.feature_engineer.free_transient_feature(&current_feature)

        # restore the best order to `samples`
        memcpy(
            self.samples + start,
            self.samples_buffer,
            sizeof(Py_ssize_t) * n_samples,
        )

        self.criterion.reset()
        self.criterion.update(start, best.split_point)
        self.criterion.child_impurity(&best.impurity_left, &best.impurity_right)
        best.impurity_improvement = self.criterion.impurity_improvement(
            parent_impurity,
            best.impurity_left,
            best.impurity_right,
            self.n_weighted_samples,
        )
        return best

    cdef void _partition_feature_buffer(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        Py_ssize_t *best_split_point,
        double *best_threshold,
        double *best_impurity,
    ) nogil:
        cdef Py_ssize_t i  # real index of samples (in `range(start, end)`)
        cdef Py_ssize_t j  # sample index (in `samples`)
        cdef Py_ssize_t p  # label index (in `labels`)
        cdef Py_ssize_t pos
        cdef Py_ssize_t new_pos
        cdef double impurity

        best_impurity[0] = -INFINITY
        best_threshold[0] = NAN
        best_split_point[0] = 0

        pos = start
        i = start
        while i < end:
            # Ignore split points with almost equal feature value
            while i + 1 < end and (
                self.feature_buffer[i + 1]
                <= self.feature_buffer[i] + FEATURE_THRESHOLD
            ):
                i += 1

            i += 1
            if i < end:
                new_pos = i
                self.criterion.update(pos, new_pos)
                pos = new_pos
                impurity = self.criterion.proxy_impurity()
                if impurity > best_impurity[0]:
                    best_impurity[0] = impurity
                    best_threshold[0] = (
                        self.feature_buffer[i - 1] / 2.0 + self.feature_buffer[i] / 2.0
                    )
                    best_split_point[0] = pos

                    if (
                        best_threshold[0] == self.feature_buffer[i]
                        or best_threshold[0] == INFINITY
                        or best_threshold[0] == -INFINITY
                    ):
                        best_threshold[0] = self.feature_buffer[i - 1]

cdef class ExtraTreeBuilder(TreeBuilder):

    cdef void _partition_feature_buffer(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        Py_ssize_t *split_point,
        double *threshold,
        double *impurity,
    ) nogil:
        cdef double min_feature = self.feature_buffer[start + 1]
        cdef double max_feature = self.feature_buffer[end - 1]
        cdef double rand_threshold = rand_uniform(
            min_feature, max_feature, &self.random_seed
        )
        cdef Py_ssize_t i

        split_point[0] = start
        for i in range(start + 1, end - 1):
            if self.feature_buffer[i] <= rand_threshold:
                split_point[0] = i
            else:
                break
        threshold[0] = rand_threshold
        impurity[0] = INFINITY


cdef struct ProximityTreePivot:
    double** data
    Py_ssize_t distance_measure
    Py_ssize_t length
    Py_ssize_t n_branches


cdef struct ProximityTreeSplit:
    Py_ssize_t* split_point
    Py_ssize_t *pivot # n_split + 1
    Py_ssize_t distance_measure
    Py_ssize_t n_split  
    double impurity


cdef void free_proximity_tree_split(ProximityTreeSplit *split) nogil:
    if split.split_point != NULL:
        free(split.split_point)
        split.split_point = NULL
    
    if split.pivot != NULL:
        free(split.pivot)
        split.pivot = NULL


cdef class ProximityTree:

    cdef Py_ssize_t _node_count
    cdef Py_ssize_t **_branches
    cdef ProximityTreePivot **_pivots
    cdef double *_values
    cdef Py_ssize_t _capacity
    cdef Py_ssize_t _n_labels
    cdef CList distance_measures

    def __cinit__(
        self, 
        list distance_measures, 
        Py_ssize_t n_labels,
        Py_ssize_t capacity=10
    ):
        self._node_count = 0
        self._capacity = capacity
        self._n_labels = n_labels

        self._branches = <Py_ssize_t**>malloc(sizeof(Py_ssize_t*) * self._n_labels)
        cdef Py_ssize_t i
        for i in range(self._n_labels):
            self._branches[i] = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * capacity)

        self._pivots = <ProximityTreePivot**> malloc(sizeof(ProximityTreePivot*) * capacity)
        self._values = <double*> malloc(sizeof(double) * capacity * n_labels)
        self.distance_measures = CList(distance_measures)
    
    cdef Py_ssize_t add_branch_node(
        self,
        Py_ssize_t parent,
        Py_ssize_t branch,
        Py_ssize_t n_samples,
        double n_weighted_samples,
        Dataset dataset,
        Py_ssize_t *pivots,
        Py_ssize_t distance_measure,
        Py_ssize_t n_split,
    ) nogil:
        cdef Py_ssize_t node_id = self._node_count
        if node_id >= self._capacity:
            if self._increase_capacity() < 0:
                return -1
        
        cdef ProximityTreePivot *pivot = <ProximityTreePivot*> malloc(sizeof(ProximityTreePivot))
        pivot.n_branches = n_split + 1
        pivot.length = dataset.n_timestep
        pivot.distance_measure = distance_measure
        pivot.data = <double**> malloc(sizeof(double*) * pivot.n_branches)
        cdef Py_ssize_t i
        for i in range(pivot.n_branches):
            pivot.data[i] = <double*> malloc(sizeof(double) * dataset.n_timestep)
            memcpy(
                pivot.data[i], 
                dataset.get_sample(pivots[i]), 
                sizeof(double) * dataset.n_timestep
            )
            
        self._pivots[node_id] = pivot
        if parent != -1:
            self._branches[branch][parent] = node_id
        self._node_count += 1
        return node_id

    cdef Py_ssize_t add_leaf_node(self, Py_ssize_t parent, Py_ssize_t branch) nogil:
        cdef Py_ssize_t node_id = self._node_count
        if node_id >= self._capacity:
            if self._increase_capacity() < 0:
                return -1
        
        if parent != -1:
            self._branches[branch][parent] = node_id
        cdef Py_ssize_t i
        for i in range(self._n_labels):
            self._branches[i][node_id] = -1
        
        self._pivots[node_id] = NULL
        self._node_count += 1
        return node_id

    cdef void set_leaf_value(
        self, 
        Py_ssize_t node_id, 
        Py_ssize_t label, 
        double value
    ) nogil:
        self._values[label + node_id * self._n_labels] = value

    cdef Py_ssize_t _increase_capacity(self) nogil:
        cdef Py_ssize_t new_capacity = self._node_count * 2
        cdef Py_ssize_t i
        for i in range(self._n_labels):
            safe_realloc(<void**> &self._branches[i], sizeof(Py_ssize_t) * new_capacity)
        safe_realloc(<void**> &self._pivots, sizeof(ProximityTreePivot*) * new_capacity)
        safe_realloc(<void**> &self._values, sizeof(double) * new_capacity * self._n_labels)
        self._capacity = new_capacity
        return 0

    @property
    def branches(self):
        branches = np.zeros((self._n_labels, self._node_count), dtype=np.intp)
        for i in range(self._n_labels):
            for j in range(self._node_count):
                branches[i, j] = self._branches[i][j]
        return branches

    @property
    def pivots(self):
        pivots = []
        cdef ProximityTreePivot *pivot
        for i in range(self._node_count):
            pivot = self._pivots[i]
            if pivot != NULL:
                arr = np.empty((pivot.n_branches, pivot.length), dtype=np.double)
                for j in range(pivot.n_branches):
                    for k in range(pivot.length):
                        arr[j, k] = self._pivots[i].data[j][k]
                pivots.append((pivot.distance_measure, arr))
            else:
                pivots.append(None)

        return pivots

    @property
    def value(self):
        cdef np.ndarray arr = np.empty(self._node_count * self._n_labels, dtype=float)
        cdef Py_ssize_t i
        for i in range(self._n_labels * self._node_count):
            arr[i] = self._values[i]
        return arr.reshape(self._node_count, self._n_labels)

    def apply(self, object x):
        if not isinstance(x, np.ndarray):
            raise ValueError("")

        x = check_dataset(x)
        cdef Dataset td = Dataset(x)
        cdef np.ndarray out = np.zeros((td.n_samples,), dtype=np.intp)
        cdef Py_ssize_t *out_data = <Py_ssize_t*> out.data
        cdef Py_ssize_t node_index
        cdef Py_ssize_t i
        cdef Py_ssize_t branch
        cdef ProximityTreePivot *pivot

        with nogil:
            for i in range(td.n_samples):
                node_index = 0
                while self._branches[0][node_index] != -1:
                    pivot = self._pivots[node_index]
                    branch = find_min_branch(
                        <DistanceMeasure> self.distance_measures.get(pivot.distance_measure),
                        pivot.data,
                        td.get_sample(i, dim=0),
                        pivot.length,
                        pivot.n_branches,
                    )
                    node_index = self._branches[branch][node_index]
                out_data[i] = node_index
        return out 

    def predict(self, object x):
        if not isinstance(x, np.ndarray):
            raise ValueError("")
        return np.take(self.value, self.apply(x), axis=0, mode="clip")


cdef Py_ssize_t find_min_branch(
    DistanceMeasure distance_measure, 
    double **pivots,
    double *sample,
    Py_ssize_t n_timestep,
    Py_ssize_t n_branches,
) nogil:
    cdef double dist
    cdef double min_dist
    cdef Py_ssize_t i
    cdef Py_ssize_t min_branch

    min_dist = INFINITY
    for i in range(n_branches):
        dist = distance_measure._distance(sample, n_timestep, pivots[i], n_timestep)
        if dist < min_dist:
            min_dist = dist
            min_branch = i
    return min_branch


cdef class ProximityTreeBuilder:

    cdef Py_ssize_t n_samples # no samples with non-zero weight
    cdef Py_ssize_t n_labels
    cdef Py_ssize_t *labels
    cdef double n_weighted_node_samples
    cdef double *weighted_label_count

    cdef Py_ssize_t max_depth
    cdef Py_ssize_t min_sample_split
    cdef Py_ssize_t min_sample_leaf
    
    cdef Py_ssize_t *pivot_buffer

    cdef double *gini_buffer
    
    cdef double *sample_weights
    cdef Py_ssize_t *samples
    cdef Py_ssize_t *samples_buffer
    cdef double *samples_branch

    cdef Py_ssize_t *label_count
    cdef Py_ssize_t *branch_count

    cdef Py_ssize_t n_features

    cdef CList distance_measures

    cdef ProximityTree tree

    cdef Dataset dataset

    cdef size_t seed

    def __cinit__(
        self, 
        np.ndarray x, 
        np.ndarray y, 
        np.ndarray sample_weights, 
        Py_ssize_t n_labels,
        size_t seed,
        Py_ssize_t n_features=1,
        Py_ssize_t max_depth=2**10,
        Py_ssize_t min_sample_split=2,
        Py_ssize_t min_sample_leaf=1,
    ):
        self.dataset = Dataset(x)
        self.labels = <Py_ssize_t*> y.data
        self.n_labels = n_labels
        self.tree = ProximityTree([_DISTANCE_MEASURE["euclidean"]()], n_labels)
        if sample_weights is None:
            self.sample_weights = NULL
        else:
            if sample_weights.dtype != np.double:
                raise ValueError("unexpected dtype (%r != np.double)" % sample_weights.dtype)
            if sample_weights.ndim != 1:
                raise ValueError("unexpected dim (%r != 1)" % sample_weights.ndim)
            if sample_weights.strides[0] // sample_weights.itemsize != 1:
                raise ValueError("unexpected stride")

            self.sample_weights = <double*> sample_weights.data
    
        self.samples = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * self.dataset.n_samples)
        self.samples_buffer = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * self.dataset.n_samples)
        self.samples_branch = <double*> malloc(sizeof(double) * self.dataset.n_samples)

        self.pivot_buffer = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * self.n_labels)
        self.label_count = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * self.n_labels)
        self.branch_count = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * self.n_labels)
        self.weighted_label_count = <double*> malloc(sizeof(double) * self.n_labels)
        self.gini_buffer = <double*> malloc(sizeof(double) * self.n_labels * self.n_labels)

        cdef Py_ssize_t i, j
        j = 0
        for i in range(self.dataset.n_samples):
            if sample_weights is None or sample_weights[i] != 0.0:
                self.samples[j] = i
                j += 1
        self.n_samples = j
        self.max_depth = max_depth
        self.n_features = n_features
        self.distance_measures = CList([_DISTANCE_MEASURE["euclidean"]()])
        self.seed = 554554354333 # seed
        self.min_sample_split = min_sample_split
        self.min_sample_leaf = min_sample_leaf

    def __dealloc__(self):
        free(self.samples)
        free(self.samples_buffer)
        free(self.pivot_buffer)
        free(self.samples_branch)
        free(self.gini_buffer)
        free(self.branch_count)

    def build_tree(self):
        with nogil:
            self._build_tree(
                0,
                self.n_samples,
                0,
                -1,
                0,
            )

    cdef void _build_tree(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        Py_ssize_t depth,
        Py_ssize_t parent,
        Py_ssize_t branch,
    ) nogil:        
        cdef Py_ssize_t i, j, current_split, split_start, node_id
        cdef Py_ssize_t n_node_samples = end - start
        cdef bint is_leaf = (
            depth >= self.max_depth
            or n_node_samples < self.min_sample_split
            or n_node_samples < 2 * self.min_sample_leaf
        )

        memset(self.weighted_label_count, 0, sizeof(double) * self.n_labels)
        self.n_weighted_node_samples = 0
        cdef double current_weight = 1.0
        for i in range(start, end):
            j = self.samples[i]
            if self.sample_weights != NULL:
                current_weight = self.sample_weights[j]

            self.n_weighted_node_samples += current_weight
            self.weighted_label_count[self.labels[j]] += current_weight

        cdef double gini = 0
        cdef double p
        for i in range(self.n_labels):
            p = self.weighted_label_count[i] / self.n_weighted_node_samples
            gini += p * (1 - p)

        if is_leaf:
            node_id = self.tree.add_leaf_node(parent, branch)
            for i in range(self.n_labels):
                self.tree.set_leaf_value(
                    node_id, 
                    i, 
                    self.weighted_label_count[i] / self.n_weighted_node_samples
                )
            return


        cdef ProximityTreeSplit split
        self._split(start, end, &split)
        is_leaf = (
            split.n_split < 1 or
            gini - split.impurity <= 0.0
        )
        if not is_leaf:
            node_id = self.tree.add_branch_node(
                parent,
                branch,
                n_node_samples,
                self.n_weighted_node_samples,
                self.dataset,
                split.pivot,
                split.distance_measure,
                split.n_split,
            )
            split_start = start
            for current_split in range(split.n_split):
                self._build_tree(split_start, split.split_point[current_split], depth + 1, node_id, current_split)
                split_start = split.split_point[current_split]

            self._build_tree(split.split_point[split.n_split - 1], end, depth + 1, node_id, split.n_split)
        else:
            node_id = self.tree.add_leaf_node(parent, branch)
            for i in range(self.n_labels):
                self.tree.set_leaf_value(
                    node_id, 
                    i, 
                    self.weighted_label_count[i] / self.n_weighted_node_samples
                )
        free_proximity_tree_split(&split)


    cdef void _split(
        self, 
        Py_ssize_t start, 
        Py_ssize_t end, 
        ProximityTreeSplit *split
    ) nogil:
        memset(self.label_count, 0, sizeof(Py_ssize_t) * self.n_labels)
        cdef Py_ssize_t i, n_branches, r, j, best_distance_measure, n_split
        cdef double gini, max_gini
        cdef Py_ssize_t n_samples = end - start
        for i in range(start, end):
            self.label_count[self.labels[self.samples[i]]] += 1
        
        n_branches = 0
        for i in range(self.n_labels):
            if self.label_count[i] > 0:
                n_branches += 1

        if n_branches < 1:
            split.n_split = 0
            split.impurity = INFINITY
            split.split_point = NULL
            split.pivot = NULL
            split.distance_measure = -1
        else:
            split.pivot = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * self.n_labels)
            split.split_point = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * self.n_labels) 
            max_gini = INFINITY
            for r in range(self.n_features):
                j = 0        
                for i in range(self.n_labels):
                    if self.label_count[i] > 0:
                        split.pivot[j] = self._sample_pivots(start, end, i)
                        j += 1
                
                split.distance_measure = self._sample_distance_measure()
                self._partition_pivots(
                    start, end, split.pivot, split.distance_measure, n_branches
                )
                gini = self._compute_gini_importance(start, end, n_branches)
                if gini < max_gini:
                    max_gini = gini
                    best_distance_measure = split.distance_measure
                    memcpy(
                        self.pivot_buffer, 
                        split.pivot, 
                        sizeof(Py_ssize_t) * self.n_labels,
                    )

            argsort(self.samples_branch + start, self.samples + start, n_samples)
            memcpy(split.pivot, self.pivot_buffer, sizeof(Py_ssize_t) * self.n_labels)
            split.distance_measure = best_distance_measure
            split.impurity = max_gini
            self._find_split_points(start, end, split)

    cdef Py_ssize_t _sample_pivots(self, Py_ssize_t start, Py_ssize_t end, Py_ssize_t label) nogil:
        label = rand_int(0, <Py_ssize_t>self.label_count[label], &self.seed)
        cdef Py_ssize_t i, j
        j = 0
        for i in range(start, end):
            if j == label:
                return self.samples[i]
            j += 1
        return -1

    cdef Py_ssize_t _sample_distance_measure(self) nogil:
        return 0

    cdef Py_ssize_t _find_split_points(
        self, 
        Py_ssize_t start, 
        Py_ssize_t end, 
        ProximityTreeSplit *split
    ) nogil:
        cdef Py_ssize_t i, current_split
        current_split = 0
        for i in range(start + 1, end):
            if self.samples_branch[i] != self.samples_branch[i - 1]:
                split.split_point[current_split] = i
                current_split += 1
        split.n_split = current_split

    cdef double _compute_gini_importance(
        self, 
        Py_ssize_t start, 
        Py_ssize_t end, 
        Py_ssize_t n_branches
    ) nogil:
        memset(self.gini_buffer, 0, sizeof(double) * self.n_labels * self.n_labels)
        cdef Py_ssize_t i, j, branch
        cdef Py_ssize_t pivot, label
        cdef double current_weight = 1.0
        for i in range(start, end):
            j = self.samples[i]
            label = self.labels[j]
            branch = (<Py_ssize_t>self.samples_branch[i])
            if self.sample_weights != NULL:
                current_weight = self.sample_weights[j]

            self.gini_buffer[label + branch * self.n_labels] += current_weight

        cdef double gini, gini_gain
        cdef double ni, p
        gini_gain = 0
        for i in range(self.n_labels):
            ni = 0
            for j in range(n_branches):
                ni += self.gini_buffer[j + i * self.n_labels]
            
            if ni > 0:
                gini = 0
                for j in range(n_branches):
                    p = self.gini_buffer[j + i * self.n_labels] / ni
                    gini += p * (1 - p)
                gini_gain += ni / self.n_weighted_node_samples * gini
        return gini_gain

    cdef void _partition_pivots(
        self, 
        Py_ssize_t start, 
        Py_ssize_t end, 
        Py_ssize_t *pivot,
        Py_ssize_t distance_measure,
        Py_ssize_t n_branches
    ) nogil:
        cdef Py_ssize_t i, j, k, min_pivots
        cdef double min_dist = INFINITY
        cdef double dist
        memset(self.branch_count, 0, sizeof(Py_ssize_t) * self.n_labels)
        for i in range(start, end):
            j = self.samples[i]
            min_dist = INFINITY
            for k in range(n_branches):
                dist = (
                    <DistanceMeasure>self.distance_measures.get(distance_measure)
                ).distance(self.dataset, pivot[k], self.dataset, j, 0)
                if dist < min_dist:
                    min_dist = dist
                    min_pivots = k
            self.samples_branch[i] = min_pivots
            self.branch_count[min_pivots] += 1

        j = 0
        for i in range(n_branches):
            if self.branch_count[i] > 0:
                pivot[j] = pivot[i]
                self.branch_count[j] = self.branch_count[i]
                j += 1
            else:
                pivot[i] = -1


def test(np.ndarray x, np.ndarray y, n_labels):
    cdef ProximityTreeBuilder b = ProximityTreeBuilder(x, y, None, n_labels, 123, max_depth=6)
    b.build_tree()
    print(b.tree.branches)
    for e in b.tree.pivots:
        print(e)
    print(b.tree.value)
    print(b.tree.apply(x))
    print(np.argmax(b.tree.predict(x), axis=1))



