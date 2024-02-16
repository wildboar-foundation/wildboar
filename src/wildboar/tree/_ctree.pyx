# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: initializedcheck=False

# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np

from libc.math cimport INFINITY, NAN, ceil, exp, fabs, log2
from libc.stdlib cimport calloc, free, malloc
from libc.string cimport memcpy, memset
from numpy cimport uint32_t

from scipy.sparse import csr_matrix

from ..distance._cdistance cimport Metric

from ..transform._attr_gen cimport Attribute, AttributeGenerator
from ..utils cimport TSArray
from ..utils._misc cimport List, argsort, safe_realloc
from ..utils._rand cimport RAND_R_MAX, rand_int, rand_uniform


cdef double ATTRIBUTE_THRESHOLD = 1e-7


cdef struct SplitPoint:
    Py_ssize_t split_point
    double threshold
    double impurity_improvement
    double impurity_left
    double impurity_right
    Attribute attribute


cdef class TreeAttributeGenerator:
    cdef AttributeGenerator generator

    def __init__(self, AttributeGenerator generator):
        self.generator = generator

    def __reduce__(self):
        return self.__class__, (self.generator, )

    cdef int reset(self, TSArray X) noexcept nogil:
        return self.generator.reset(X)

    cdef Py_ssize_t get_n_attributess(self, TSArray X, Py_ssize_t depth) noexcept nogil:
        return self.generator.get_n_attributess(X)

    cdef Py_ssize_t next_attribute(
        self,
        Py_ssize_t attribute_id,
        TSArray X,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        Attribute *transient,
        uint32_t *seed
    ) noexcept nogil:
        return self.generator.next_attribute(
            attribute_id, X, samples, n_samples, transient, seed
        )

    cdef Py_ssize_t init_persistent(
        self,
        TSArray X,
        Attribute *transient,
        Attribute *persistent
    ) noexcept nogil:
        return self.generator.init_persistent(X, transient, persistent)

    cdef Py_ssize_t free_transient(self, Attribute *attribute) noexcept nogil:
        return self.generator.free_transient(attribute)

    cdef Py_ssize_t free_persistent(self, Attribute *attribute) noexcept nogil:
        return self.generator.free_persistent(attribute)

    cdef double transient_value(
        self,
        Attribute *attribute,
        TSArray X,
        Py_ssize_t sample
    ) noexcept nogil:
        return self.generator.transient_value(attribute, X, sample)

    cdef double persistent_value(
        self,
        Attribute *attribute,
        TSArray X,
        Py_ssize_t sample
    ) noexcept nogil:
        return self.generator.persistent_value(attribute, X, sample)

    cdef void transient_values(
        self,
        Attribute *attribute,
        TSArray X,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        double* values
    ) noexcept nogil:
        self.generator.transient_values(
            attribute, X, samples, n_samples, values
        )

    cdef void persistent_values(
        self,
        Attribute *attribute,
        TSArray X,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        double* values
    ) noexcept nogil:
        self.generator.persistent_values(
            attribute, X, samples, n_samples, values
        )

    cdef object persistent_to_object(self, Attribute *attribute):
        return self.generator.persistent_to_object(attribute)

    cdef Py_ssize_t persistent_from_object(self, object object, Attribute *attribute):
        return self.generator.persistent_from_object(object, attribute)


cdef class DynamicTreeAttributeGenerator(TreeAttributeGenerator):

    cdef double alpha

    def __init__(self, AttributeGenerator generator, double alpha):
        super().__init__(generator)
        self.alpha = alpha

    def __reduce__(self):
        return self.__class__, (self.generator, self.alpha)

    cdef Py_ssize_t get_n_attributess(self, TSArray X, Py_ssize_t depth) noexcept nogil:
        cdef Py_ssize_t n_attributes = self.generator.get_n_attributess(X)
        cdef double weight = 1.0 - exp(-fabs(self.alpha) * depth)
        if self.alpha < 0:
            weight = 1 - weight

        return <Py_ssize_t> max(1, ceil(n_attributes * weight))


cdef class Criterion:

    cdef double weighted_n_left
    cdef double weighted_n_right
    cdef double weighted_n_total
    cdef Py_ssize_t start
    cdef Py_ssize_t end
    cdef Py_ssize_t *samples
    cdef const double[:] sample_weight

    cdef void init(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        Py_ssize_t *samples,
        const double[:] sample_weights,
    ) noexcept nogil:
        self.start = start
        self.end = end
        self.samples = samples
        self.sample_weight = sample_weights

    cdef void reset(self) noexcept nogil:
        pass

    cdef void update(self, Py_ssize_t pos, Py_ssize_t new_pos) noexcept nogil:
        pass

    cdef double proxy_impurity(self) noexcept nogil:
        cdef double left_impurity
        cdef double right_impurity
        self.child_impurity(&left_impurity, &right_impurity)
        return -self.weighted_n_right * right_impurity - self.weighted_n_left * left_impurity

    cdef double impurity(self) noexcept nogil:
        pass

    cdef void child_impurity(self, double *left, double *right) noexcept nogil:
        pass

    cdef double impurity_improvement(
        self,
        double impurity_parent,
        double impurity_left,
        double impurity_right,
        double weighted_n_samples,
    ) noexcept nogil:
        return ((self.weighted_n_total / weighted_n_samples) *
                (impurity_parent - (self.weighted_n_right / self.weighted_n_total * impurity_right)
                                 - (self.weighted_n_left / self.weighted_n_total * impurity_left)))

    cdef void leaf_value(self, Tree tree, Py_ssize_t node_id) noexcept nogil:
        pass

cdef class ClassificationCriterion(Criterion):

    cdef const Py_ssize_t[:] labels
    cdef Py_ssize_t n_labels
    cdef double *sum_left
    cdef double *sum_right
    cdef double *sum_total

    def __cinit__(self, const Py_ssize_t[:] y, Py_ssize_t n_labels):
        self.labels = y
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
        double[:] sample_weights,
    ) noexcept nogil:
        Criterion.init(self, start, end, samples, sample_weights)
        self.weighted_n_total = 0

        memset(self.sum_total, 0, self.n_labels * sizeof(double))

        cdef Py_ssize_t i, j
        cdef double w = 1.0
        for i in range(start, end):
            j = samples[i]
            if sample_weights is not None:
                w = sample_weights[j]

            self.sum_total[self.labels[j]] += w
            self.weighted_n_total += w

        self.reset()

    cdef void reset(self) noexcept nogil:
        self.weighted_n_left = 0
        self.weighted_n_right = self.weighted_n_total
        memset(self.sum_left, 0, self.n_labels * sizeof(double))
        memcpy(self.sum_right, self.sum_total, self.n_labels * sizeof(double))

    cdef void update(self, Py_ssize_t pos, Py_ssize_t new_pos) noexcept nogil:
        cdef Py_ssize_t i, j
        cdef double w = 1.0

        for i in range(pos, new_pos):
            j = self.samples[i]
            if self.sample_weight is not None:
                w = self.sample_weight[j]

            self.sum_left[self.labels[j]] += w
            self.weighted_n_left += w

        self.weighted_n_right = self.weighted_n_total - self.weighted_n_left
        for i in range(self.n_labels):
            self.sum_right[i] = self.sum_total[i] - self.sum_left[i]

    cdef double impurity(self) noexcept nogil:
        pass

    cdef void child_impurity(self, double* left, double *right) noexcept nogil:
        pass

    cdef void leaf_value(self, Tree tree, Py_ssize_t node_id) noexcept nogil:
        cdef Py_ssize_t i
        cdef double prob
        for i in range(self.n_labels):
            prob = self.sum_total[i] / self.weighted_n_total
            tree.set_leaf_value(node_id, i, prob)

cdef class GiniCriterion(ClassificationCriterion):

    cdef double impurity(self) noexcept nogil:
        cdef double sq_count = 0.0
        cdef double c
        cdef Py_ssize_t i
        for i in range(self.n_labels):
            c = self.sum_total[i]
            sq_count += c * c

        return 1.0 - sq_count / (self.weighted_n_total * self.weighted_n_total)

    cdef void child_impurity(self, double *left, double *right) noexcept nogil:
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

    cdef double impurity(self) noexcept nogil:
        cdef double c
        cdef double entropy = 0
        cdef Py_ssize_t i
        for i in range(self.n_labels):
            c = self.sum_total[i]
            if c > 0:
                c /= self.weighted_n_total
                entropy -= c * log2(c)

        return entropy

    cdef void child_impurity(self, double *left, double *right) noexcept nogil:
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
    cdef const double[:] labels
    cdef Py_ssize_t pos

    def __cinit__(self, const double[:] y):
        self.labels = y
        self.sum_left = 0
        self.sum_right = 0
        self.sum_total = 0
        self.sum_sq_total = 0

    cdef void init(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        Py_ssize_t *samples,
        double[:] sample_weights,
    ) noexcept nogil:
        Criterion.init(self, start, end, samples, sample_weights)
        self.sum_total = 0
        self.sum_sq_total = 0
        self.weighted_n_total = 0

        cdef Py_ssize_t i, j
        cdef double x
        cdef double w = 1.0

        for i in range(start, end):
            j = samples[i]
            if sample_weights is not None:
                w = sample_weights[j]

            x = w * self.labels[j]
            self.sum_total += x
            self.sum_sq_total += x * x
            self.weighted_n_total += w

        self.reset()
        self.start = start

    cdef void reset(self) noexcept nogil:
        self.weighted_n_left = 0
        self.weighted_n_right = self.weighted_n_total
        self.sum_left = 0
        self.sum_right = self.sum_total
        self.pos = 0

    cdef void update(self, Py_ssize_t pos, Py_ssize_t new_pos) noexcept nogil:
        cdef Py_ssize_t i
        cdef Py_ssize_t j
        cdef double w = 1.0
        for i in range(pos, new_pos):
            j = self.samples[i]
            if self.sample_weight is not None:
                w = self.sample_weight[j]

            self.sum_left += w * self.labels[j]
            self.weighted_n_left += w

        self.weighted_n_right = self.weighted_n_total - self.weighted_n_left
        self.sum_right = self.sum_total - self.sum_left
        self.pos = new_pos

    cdef void leaf_value(self, Tree tree, Py_ssize_t node_id) noexcept nogil:
        tree.set_leaf_value(node_id, 0, self.sum_total / self.weighted_n_total)

cdef class MSECriterion(RegressionCriterion):

    cdef double proxy_impurity(self) noexcept nogil:
        cdef double proxy_impurity_left = self.sum_left * self.sum_left
        cdef double proxy_impurity_right = self.sum_right * self.sum_right
        return proxy_impurity_left / self.weighted_n_left + proxy_impurity_right / self.weighted_n_right

    cdef double impurity(self) noexcept nogil:
        cdef double impurity
        impurity = self.sum_sq_total / self.weighted_n_total
        impurity -= (self.sum_total / self.weighted_n_total) ** 2
        return impurity

    cdef void child_impurity(self, double* left, double *right) noexcept nogil:
        cdef double left_sq_sum = 0
        cdef double right_sq_sum = 0
        cdef double w = 1.0
        cdef double y
        cdef Py_ssize_t i, j

        for i in range(self.start, self.pos):
            j = self.samples[i]
            if self.sample_weight is not None:
                w = self.sample_weight[j]

            y = self.labels[j]
            left_sq_sum += w * y * y
        right_sq_sum = self.sum_sq_total - left_sq_sum

        left[0] = left_sq_sum / self.weighted_n_left
        left[0] -= (self.sum_left / self.weighted_n_left) ** 2
        right[0] = right_sq_sum / self.weighted_n_right
        right[0] -= (self.sum_right / self.weighted_n_right) ** 2

cpdef Tree _make_tree(
    TreeAttributeGenerator generator,
    Py_ssize_t n_labels,
    Py_ssize_t max_depth,
    list attributes,
    object threshold,
    object value,
    object left,
    object right,
    object impurity,
    object n_node_samples,
    object n_weighted_node_samples
):
    cdef Tree tree = Tree(generator, n_labels, capacity=len(attributes) + 1)
    tree._max_depth = max_depth
    cdef Py_ssize_t node_count = len(attributes)
    cdef Py_ssize_t i
    cdef Py_ssize_t dim
    cdef object arr
    cdef Attribute *attribute
    cdef object value_reshape = value.reshape(-1)

    tree._node_count = node_count
    for i in range(node_count):
        if attributes[i] is not None:
            attribute = <Attribute*> malloc(sizeof(Attribute))
            generator.persistent_from_object(attributes[i], attribute)
            tree._attributes[i] = attribute
        else:
            tree._attributes[i] = NULL
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

    cdef TreeAttributeGenerator generator

    cdef Py_ssize_t _max_depth
    cdef Py_ssize_t _capacity
    cdef Py_ssize_t _n_labels  # 1 for regression

    cdef Py_ssize_t _node_count
    cdef Py_ssize_t *_left
    cdef Py_ssize_t *_right
    cdef Attribute **_attributes
    cdef double *_thresholds
    cdef double *_impurity
    cdef double *_values
    cdef double *_n_weighted_node_samples
    cdef Py_ssize_t *_n_node_samples

    def __cinit__(
        self,
        TreeAttributeGenerator generator,
        Py_ssize_t n_labels,
        Py_ssize_t capacity=10
    ):
        self.generator = generator
        self._node_count = 0
        self._capacity = capacity
        self._n_labels = n_labels
        self._attributes = <Attribute**> malloc(self._capacity * sizeof(Attribute*))
        self._thresholds = <double*> malloc(self._capacity * sizeof(double))
        self._values = <double*> malloc(self._capacity * self._n_labels * sizeof(double))
        self._left = <Py_ssize_t*> malloc(self._capacity * sizeof(Py_ssize_t))
        self._right = <Py_ssize_t*> malloc(self._capacity * sizeof(Py_ssize_t))
        self._impurity = <double*> malloc(self._capacity * sizeof(double))
        self._n_node_samples = <Py_ssize_t*> malloc(self._capacity * sizeof(Py_ssize_t))
        self._n_weighted_node_samples = <double*> malloc(self._capacity * sizeof(double))

    def __dealloc__(self):
        cdef Py_ssize_t i
        if self._attributes != NULL:
            for i in range(self._node_count):
                if self._attributes[i] != NULL:
                    self.generator.free_persistent(self._attributes[i])
                    free(self._attributes[i])
            free(self._attributes)

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
            self.generator,
            self._n_labels,
            self._max_depth,
            self.attribute,
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
        cdef object arr = np.empty(self._node_count * self._n_labels, dtype=float)
        cdef Py_ssize_t i
        for i in range(self._n_labels * self._node_count):
            arr[i] = self._values[i]
        return arr.reshape(self._node_count, self._n_labels)

    @property
    def attribute(self):
        cdef Py_ssize_t i, j
        cdef Attribute* attribute
        cdef object object
        cdef list ret = []
        for i in range(self._node_count):
            attribute = self._attributes[i]
            if attribute != NULL:
                object = self.generator.persistent_to_object(attribute)
                ret.append(object)
            else:
                ret.append(None)
        return ret

    @property
    def feature(self):
        import warnings
        warnings.warn(
            "`feature` has been renamed to `attribute` in 1.2 and will be removed in 1.4",
            DeprecationWarning,
        )
        return self.attribute

    @property
    def n_node_samples(self):
        cdef object arr = np.zeros(self._node_count, dtype=np.intp)
        cdef Py_ssize_t i
        for i in range(self._node_count):
            arr[i] = self._n_node_samples[i]
        return arr

    @property
    def n_weighted_node_samples(self):
        cdef object arr = np.zeros(self._node_count, dtype=float)
        cdef Py_ssize_t i
        for i in range(self._node_count):
            arr[i] = self._n_weighted_node_samples[i]
        return arr

    @property
    def left(self):
        cdef object arr = np.empty(self._node_count, dtype=np.intp)
        cdef Py_ssize_t i
        for i in range(self._node_count):
            arr[i] = self._left[i]
        return arr

    @property
    def right(self):
        cdef object arr = np.empty(self._node_count, dtype=np.intp)
        cdef Py_ssize_t i
        for i in range(self._node_count):
            arr[i] = self._right[i]
        return arr

    @property
    def threshold(self):
        cdef object arr = np.empty(self._node_count, dtype=float)
        cdef Py_ssize_t i
        for i in range(self._node_count):
            arr[i] = self._thresholds[i]
        return arr

    @property
    def impurity(self):
        cdef object arr = np.empty(self._node_count, dtype=float)
        cdef Py_ssize_t i
        for i in range(self._node_count):
            arr[i] = self._impurity[i]
        return arr

    def predict(self, object X):
        cdef object apply = self.apply(X)
        return np.take(self.value, apply, axis=0, mode="clip")

    def apply(self, TSArray X):
        cdef Py_ssize_t[:] out = np.zeros((X.shape[0],), dtype=np.intp)
        cdef Attribute *attribute
        cdef double threshold, attribute_value
        cdef Py_ssize_t node_index
        cdef Py_ssize_t i
        with nogil:
            self.generator.reset(X)
            for i in range(X.shape[0]):
                node_index = 0
                while self._left[node_index] != -1:
                    threshold = self._thresholds[node_index]
                    attribute = self._attributes[node_index]
                    attribute_value = self.generator.persistent_value(
                        attribute, X, i
                    )
                    if attribute_value <= threshold:
                        node_index = self._left[node_index]
                    else:
                        node_index = self._right[node_index]

                out[i] = <Py_ssize_t> node_index

        return out.base

    def decision_path(self, TSArray X):
        cdef Py_ssize_t[:, :] out = np.zeros((X.shape[0], self.node_count), dtype=np.intp)
        cdef Py_ssize_t node_index
        cdef Py_ssize_t i
        cdef Attribute *attribute
        cdef double threshold, attribute_value
        with nogil:
            self.generator.reset(X)
            for i in range(X.shape[0]):
                node_index = 0
                while self._left[node_index] != -1:
                    out[i, node_index] = 1
                    threshold = self._thresholds[node_index]
                    attribute = self._attributes[node_index]
                    attribute_value = self.generator.persistent_value(
                        attribute, X, i
                    )
                    if attribute_value <= threshold:
                        node_index = self._left[node_index]
                    else:
                        node_index = self._right[node_index]

        return csr_matrix(out.base, dtype=bool)

    @property
    def node_count(self):
        return self._node_count

    cdef Py_ssize_t add_leaf_node(
        self,
        Py_ssize_t parent,
        bint is_left,
        Py_ssize_t n_node_samples,
        double n_weighted_node_samples,
    ) noexcept nogil:
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
        self._attributes[node_id] = NULL
        self._node_count += 1
        return node_id

    cdef void set_leaf_value(
        self,
        Py_ssize_t node_id,
        Py_ssize_t out_label,
        double out_value,
    ) noexcept nogil:
        self._values[out_label + node_id * self._n_labels] = out_value

    cdef Py_ssize_t add_branch_node(
        self,
        Py_ssize_t parent,
        bint is_left,
        Py_ssize_t n_node_samples,
        double n_weighted_node_samples,
        Attribute *attribute,
        double threshold,
        double impurity,
    ) noexcept nogil:
        cdef Py_ssize_t node_id = self._node_count
        if node_id >= self._capacity:
            if self._increase_capacity() == -1:
                return -1

        self._impurity[node_id] = impurity
        self._n_node_samples[node_id] = n_node_samples
        self._n_weighted_node_samples[node_id] = n_weighted_node_samples
        self._thresholds[node_id] = threshold
        self._attributes[node_id] = attribute
        if parent != -1:
            if is_left:
                self._left[parent] = node_id
            else:
                self._right[parent] = node_id

        self._node_count += 1
        return node_id

    cdef Py_ssize_t _increase_capacity(self) noexcept nogil:
        cdef Py_ssize_t new_capacity = self._node_count * 2
        cdef Py_ssize_t ret
        ret = safe_realloc(<void**> &self._attributes, sizeof(Attribute*) * new_capacity)
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

    # the weight of the j:th sample
    cdef const double[:] sample_weights

    # the dataset of time series
    cdef TSArray X

    # the number of samples with non-zero weight
    cdef Py_ssize_t n_samples
    cdef double n_weighted_samples

    # buffer of samples from 0, ..., n_samples
    cdef Py_ssize_t *samples

    # temporary buffer of samples from 0, ..., n_samples
    cdef Py_ssize_t *samples_buffer

    # temporary buffer for attribute computations
    cdef double *attribute_buffer

    cdef TreeAttributeGenerator generator
    cdef Criterion criterion
    cdef Tree tree
    cdef uint32_t random_seed

    def __cinit__(
        self,
        TSArray X,
        const double[:] sample_weights,
        TreeAttributeGenerator generator,
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

        self.X = X
        self.generator = generator
        self.criterion = criterion
        self.tree = tree

        self.current_node_id = 0
        self.n_samples = self.X.shape[0]
        self.samples = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * self.n_samples)
        self.samples_buffer = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * self.n_samples)
        self.attribute_buffer = <double*> malloc(sizeof(double) * self.n_samples)

        if (
            self.samples == NULL or
            self.attribute_buffer == NULL or
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
        self.sample_weights = sample_weights

    def __dealloc__(self):
        free(self.samples)
        free(self.samples_buffer)
        free(self.attribute_buffer)

    @property
    def tree_(self):
        return self.tree

    cpdef Py_ssize_t build_tree(self):
        cdef Py_ssize_t root_node_id
        cdef Py_ssize_t max_depth = 0
        with nogil:
            self.generator.reset(self.X)
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
    ) noexcept nogil:
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
        Attribute *persistent,
        Py_ssize_t parent,
        bint is_left,
    ) noexcept nogil:
        cdef Py_ssize_t node_id
        node_id = self.tree.add_branch_node(
            parent,
            is_left,
            end - start,
            self.criterion.weighted_n_total,
            persistent,
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
    ) noexcept nogil:
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

        cdef SplitPoint split = self._split(start, end, depth, impurity)

        cdef Attribute *persistent
        cdef Py_ssize_t current_node_id
        cdef Py_ssize_t err
        is_leaf = (
            split.split_point <= start
            or split.split_point >= end
            or split.impurity_improvement <= self.min_impurity_decrease
        )
        if not is_leaf:
            # The persistent attribute is freed by the Tree
            persistent = <Attribute*> malloc(sizeof(Attribute))
            err = self.generator.init_persistent(
                self.X, &split.attribute, persistent
            )
            self.generator.free_transient(&split.attribute)
            if err == -1:
                return -1

            current_node_id = self.new_branch_node(
                start, end, split, persistent, parent, is_left
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

    cdef SplitPoint _split(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        Py_ssize_t depth,
        double parent_impurity
    ) noexcept nogil:
        cdef Py_ssize_t i, n_samples
        cdef Py_ssize_t current_split_point
        cdef double current_threshold
        cdef double current_impurity
        cdef double best_impurity
        cdef Attribute current_attribute

        n_samples = end - start

        best_impurity = -INFINITY

        current_attribute.attribute = NULL
        current_impurity = -INFINITY
        current_threshold = NAN
        current_split_point = 0

        cdef SplitPoint best
        best.threshold = NAN
        best.split_point = 0
        best.attribute.attribute = NULL

        for i in range(self.generator.get_n_attributess(self.X, depth)):
            self.generator.next_attribute(
                i,
                self.X,
                self.samples + start,
                n_samples,
                &current_attribute,
                &self.random_seed
            )

            self.generator.transient_values(
                &current_attribute,
                self.X,
                self.samples + start,
                end - start,
                self.attribute_buffer + start,
            )
            argsort(self.attribute_buffer + start, self.samples + start, n_samples)

            # All attribute values are constant
            if self.attribute_buffer[end - 1] <= self.attribute_buffer[start] + ATTRIBUTE_THRESHOLD:
                continue

            self.criterion.reset()
            self._partition_attribute_buffer(
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
                if best.attribute.attribute != NULL:
                    self.generator.free_transient(&best.attribute)

                best.attribute = current_attribute
            else:
                self.generator.free_transient(&current_attribute)

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

    cdef void _partition_attribute_buffer(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        Py_ssize_t *best_split_point,
        double *best_threshold,
        double *best_impurity,
    ) noexcept nogil:
        cdef Py_ssize_t i  # real index of samples (in `range(start, end)`)
        cdef Py_ssize_t j  # sample index (in `samples`)
        cdef Py_ssize_t pos
        cdef Py_ssize_t new_pos
        cdef double impurity

        best_impurity[0] = -INFINITY
        best_threshold[0] = NAN
        best_split_point[0] = 0

        pos = start
        i = start
        while i < end:
            # Ignore split points with almost equal attribute value
            while i + 1 < end and (
                self.attribute_buffer[i + 1]
                <= self.attribute_buffer[i] + ATTRIBUTE_THRESHOLD
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
                        self.attribute_buffer[i - 1] / 2.0 + self.attribute_buffer[i] / 2.0
                    )
                    best_split_point[0] = pos

                    if (
                        best_threshold[0] == self.attribute_buffer[i]
                        or best_threshold[0] == INFINITY
                        or best_threshold[0] == -INFINITY
                    ):
                        best_threshold[0] = self.attribute_buffer[i - 1]

cdef class ExtraTreeBuilder(TreeBuilder):

    cdef void _partition_attribute_buffer(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        Py_ssize_t *split_point,
        double *threshold,
        double *impurity,
    ) noexcept nogil:
        cdef double min_attribute = self.attribute_buffer[start + 1]
        cdef double max_attribute = self.attribute_buffer[end - 1]
        cdef double rand_threshold = rand_uniform(
            min_attribute, max_attribute, &self.random_seed
        )
        cdef Py_ssize_t i

        split_point[0] = start
        for i in range(start + 1, end - 1):
            if self.attribute_buffer[i] <= rand_threshold:
                split_point[0] = i
            else:
                break
        threshold[0] = rand_threshold
        impurity[0] = INFINITY
