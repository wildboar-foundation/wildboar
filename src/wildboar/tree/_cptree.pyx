# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np

cimport numpy as np
from libc.math cimport INFINITY, NAN, log2
from libc.stdlib cimport calloc, free, malloc
from libc.string cimport memcpy, memset

from ..distance._distance cimport DistanceMeasure

from ..utils.data import check_dataset
from ..utils.data cimport Dataset
from ..utils.misc cimport CList, argsort, safe_realloc
from ..utils.rand cimport (
    RAND_R_MAX,
    VoseRand,
    rand_int,
    vose_rand_free,
    vose_rand_init,
    vose_rand_int,
    vose_rand_precompute,
)


cdef struct Pivot:
    double** data
    Py_ssize_t distance_measure
    Py_ssize_t length
    Py_ssize_t n_branches


cdef struct Split:
    Py_ssize_t* split_point # n_split
    Py_ssize_t *pivot # n_split + 1
    Py_ssize_t distance_measure
    Py_ssize_t n_split
    double impurity_improvement
    double *child_impurity # n_split + 1


cdef void free_split(Split *split) nogil:
    if split.split_point != NULL:
        free(split.split_point)
        split.split_point = NULL

    if split.pivot != NULL:
        free(split.pivot)
        split.pivot = NULL

    if split.child_impurity != NULL:
        free(split.child_impurity)
        split.child_impurity = NULL

cpdef Tree _make_tree(
    list distance_measures,
    Py_ssize_t n_labels,
    Py_ssize_t max_depth,
    np.ndarray branch,
    list pivots,
    np.ndarray value,
):
    cdef Tree tree = Tree(distance_measures, n_labels, capacity=len(pivots) + 1)
    tree._max_depth = max_depth
    tree._node_count = len(pivots)
    cdef Py_ssize_t i, j, k
    cdef Pivot *pivot
    cdef np.ndarray arr
    value = value.reshape(-1)
    for i in range(tree._node_count):
        if pivots[i] is not None:
            pivot = <Pivot*> malloc(sizeof(Pivot))
            pivot.distance_measure = pivots[i][0]
            arr = pivots[i][1]
            pivot.n_branches = arr.shape[0]
            pivot.length = arr.shape[1]
            pivot.data = <double**> malloc(sizeof(double*) * pivot.n_branches)
            for j in range(pivot.n_branches):
                pivot.data[j] = <double*> malloc(sizeof(double) * pivot.length)
                for k in range(pivot.length):
                    pivot.data[j][k] = arr[j, k]

            tree._pivots[i] = pivot
        else:
            tree._pivots[i] = NULL

        for j in range(n_labels):
            tree._branches[j][i] = branch[j, i]

    for i in range(tree._node_count * n_labels):
        tree._values[i] = value[i]

    return tree


cdef class Tree:

    cdef Py_ssize_t _max_depth
    cdef Py_ssize_t _node_count
    cdef Py_ssize_t **_branches
    cdef Pivot **_pivots
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

        self._pivots = <Pivot**> malloc(sizeof(Pivot*) * capacity)
        self._values = <double*> malloc(sizeof(double) * capacity * n_labels)
        self.distance_measures = CList(distance_measures)

    def __reduce__(self):
       return _make_tree, (
           self.distance_measures.py_list,
           self._n_labels,
           self._max_depth,
           self.branch,
           self.pivot,
           self.value,
       )

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

        cdef Pivot *pivot = <Pivot*> malloc(sizeof(Pivot))
        pivot.n_branches = n_split + 1
        pivot.length = dataset.n_timestep
        pivot.distance_measure = distance_measure
        pivot.data = <double**> malloc(sizeof(double*) * pivot.n_branches)
        cdef Py_ssize_t i
        for i in range(pivot.n_branches):
            pivot.data[i] = <double*> malloc(sizeof(double) * dataset.n_timestep)
            memcpy(
                pivot.data[i],
                dataset.get_sample(pivots[i], 0),
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
        safe_realloc(<void**> &self._pivots, sizeof(Pivot*) * new_capacity)
        safe_realloc(<void**> &self._values, sizeof(double) * new_capacity * self._n_labels)
        self._capacity = new_capacity
        return 0

    @property
    def branch(self):
        branches = np.zeros((self._n_labels, self._node_count), dtype=np.intp)
        for i in range(self._n_labels):
            for j in range(self._node_count):
                branches[i, j] = self._branches[i][j]
        return branches

    @property
    def pivot(self):
        pivots = []
        cdef Pivot *pivot
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

    @property
    def max_depth(self):
        return self._max_depth

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
        cdef Pivot *pivot

        with nogil:
            for i in range(self.distance_measures.size):
                (<DistanceMeasure>self.distance_measures.get(i)).reset(td, td)

            for i in range(td.n_samples):
                node_index = 0
                while self._branches[0][node_index] != -1:
                    pivot = self._pivots[node_index]
                    branch = find_min_branch(
                        <DistanceMeasure> self.distance_measures.get(pivot.distance_measure),
                        pivot.data,
                        td.get_sample(i, 0),
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


cdef class Criterion:

    cdef Py_ssize_t *labels
    cdef Py_ssize_t label_stride

    cdef Py_ssize_t n_labels
    cdef double weighted_n_total
    cdef double *weighted_n_branch
    cdef double *weighted_label_count
    cdef double *weighted_label_branch_count
    cdef Py_ssize_t *label_count

    cdef Py_ssize_t start
    cdef Py_ssize_t end
    cdef Py_ssize_t *samples
    cdef double *sample_weight

    def __cinit__(self, np.ndarray y, Py_ssize_t n_labels):
        if y.dtype != np.intp:
            raise ValueError("unexpected dtype (%r != %r)" % (y.dtype, np.intp))

        if y.ndim != 1:
            raise ValueError("unexpected dim (%r != 1)" % y.ndim)

        self.labels = <Py_ssize_t*> y.data
        self.label_stride = <Py_ssize_t> y.strides[0] / <Py_ssize_t> y.itemsize
        self.n_labels = n_labels
        self.weighted_label_count = <double*> calloc(n_labels, sizeof(double))
        self.weighted_n_branch = <double*> calloc(n_labels, sizeof(double))
        self.weighted_label_branch_count = <double*> calloc(
            n_labels * n_labels, sizeof(double)
        )
        self.label_count = <Py_ssize_t*> calloc(n_labels, sizeof(Py_ssize_t))

    cdef void init(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        Py_ssize_t *samples,
        double *sample_weight,
    ) nogil:
        self.start = start
        self.end = end
        self.samples = samples
        self.sample_weight = sample_weight

        memset(self.weighted_label_count, 0, sizeof(double) * self.n_labels)
        memset(self.label_count, 0, sizeof(Py_ssize_t) * self.n_labels)
        self.weighted_n_total = 0
        cdef Py_ssize_t i, j, p
        cdef double w = 1.0
        for i in range(start, end):
            j = self.samples[i]
            p = j * self.label_stride
            if self.sample_weight != NULL:
                w = self.sample_weight[j]

            self.weighted_n_total += w
            self.weighted_label_count[self.labels[p]] += w
            self.label_count[self.labels[p]] += 1

    cdef void reset(self, double *samples_branch) nogil:
        cdef Py_ssize_t i, j, p
        cdef Py_ssize_t label, branch
        cdef double w = 1.0
        memset(
            self.weighted_label_branch_count,
            0,
            sizeof(double) * self.n_labels * self.n_labels
        )
        memset(self.weighted_n_branch, 0, sizeof(double) * self.n_labels)
        for i in range(self.start, self.end):
            j = self.samples[i]
            p = j * self.label_stride
            label = self.labels[p]
            branch = (<Py_ssize_t>samples_branch[i])
            if self.sample_weight != NULL:
                w = self.sample_weight[j]

            self.weighted_n_branch[branch] += w
            self.weighted_label_branch_count[label + branch * self.n_labels] += w

    cdef double impurity(self) nogil:
        return NAN

    cdef void child_impurity(self, double *branches, Py_ssize_t n_branches) nogil:
        pass

    cdef double proxy_impurity(self, double* branches, Py_ssize_t n_branches) nogil:
        self.child_impurity(branches, n_branches)
        cdef double impurity = 0.0
        for i in range(n_branches):
            impurity -= self.weighted_n_branch[i] * branches[i]
        return impurity

    cdef double impurity_improvement(
        self,
        double impurity_parent,
        double *child_impurity,
        Py_ssize_t n_branches,
        double n_weighted_samples,
    ) nogil:
        cdef Py_ssize_t i
        for i in range(n_branches):
            impurity_parent -= self.weighted_n_branch[i] / self.weighted_n_total * child_impurity[i]
        return self.weighted_n_total / n_weighted_samples * impurity_parent


cdef class GiniCriterion(Criterion):

    cdef double impurity(self) nogil:
        cdef double sq_count = 0.0
        cdef double c
        cdef Py_ssize_t i
        for i in range(self.n_labels):
            c = self.weighted_label_count[i]
            sq_count += c * c
        return 1.0 - sq_count / (self.weighted_n_total * self.weighted_n_total)

    cdef void child_impurity(self, double *branches, Py_ssize_t n_branches) nogil:
        cdef Py_ssize_t label, branch
        cdef double v
        for branch in range(n_branches):
            branches[branch] = 0.0
            for label in range(self.n_labels):
                v = self.weighted_label_branch_count[label + branch * self.n_labels]
                branches[branch] += v * v

        for branch in range(n_branches):
            if self.weighted_n_branch[branch] > 0:
                branches[branch] = 1 - branches[branch] / (
                    self.weighted_n_branch[branch] * self.weighted_n_branch[branch]
                )


cdef class EntropyCriterion(Criterion):

    cdef double impurity(self) nogil:
        cdef double c
        cdef double entropy = 0
        cdef Py_ssize_t i
        for i in range(self.n_labels):
            c = self.weighted_label_count[i]
            if c > 0:
                c /= self.weighted_n_total
                entropy -= c * log2(c)
        return entropy

    cdef void child_impurity(self, double *branches, Py_ssize_t n_branches) nogil:
        cdef double v
        cdef Py_ssize_t label, branch
        for branch in range(n_branches):
            branches[branch] = 0
            for label in range(self.n_labels):
                v = self.weighted_label_branch_count[label + branch * self.n_labels]
                if v > 0:
                    v /= self.weighted_n_branch[branch]
                    branches[branch] -= v * log2(v)


cdef class DistanceMeasureSampler:
    cdef VoseRand vr
    cdef Py_ssize_t n_measures

    def __cinit__(self, Py_ssize_t n_measures, np.ndarray weights=None):
        self.n_measures = n_measures
        if weights is not None:
            if weights.dtype != np.double:
                raise ValueError("unexpected dtype (%r != np.double)" % weights.dtype)
            if weights.ndim != 1:
                raise ValueError("unexpected dim (%r != 1)" % weights.ndim)
            if weights.strides[0] // weights.itemsize != 1:
                raise ValueError("unexpected stride")
            vose_rand_init(&self.vr, weights.size)
            vose_rand_precompute(&self.vr, <double*> weights.data)

    def __dealloc__(self):
        vose_rand_free(&self.vr)

    cdef Py_ssize_t sample(
        self,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        size_t *seed
    ) nogil:
        pass

cdef class UniformDistanceMeasureSampler(DistanceMeasureSampler):

    cdef Py_ssize_t sample(
        self,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        size_t *seed
    ) nogil:
        return rand_int(0, self.n_measures, seed)


cdef class WeightedDistanceMeasureSampler(DistanceMeasureSampler):

    cdef Py_ssize_t sample(
        self,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        size_t *seed
    ) nogil:
        return vose_rand_int(&self.vr, seed)


cdef class PivotSampler:

    cdef Py_ssize_t sample(
        self,
        Py_ssize_t *labels,
        Py_ssize_t label_stride,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        Py_ssize_t label,
        Py_ssize_t *label_count,
        size_t *seed,
    ) nogil:
        pass


cdef class LabelPivotSampler(PivotSampler):

    cdef Py_ssize_t sample(
        self,
        Py_ssize_t *labels,
        Py_ssize_t label_stride,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        Py_ssize_t label,
        Py_ssize_t *label_count,
        size_t *seed,
    ) nogil:
        cdef Py_ssize_t n_labels = rand_int(0, label_count[label], seed)
        cdef Py_ssize_t i, p
        cdef Py_ssize_t label_index = 0
        for i in range(n_samples):
            p = samples[i] * label_stride
            if label_index == n_labels:
                return samples[i]

            if labels[p] == label:
                label_index += 1

        # This code is unreachable
        return -1

cdef class UniformPivotSampler(PivotSampler):

    cdef Py_ssize_t sample(
        self,
        Py_ssize_t *labels,
        Py_ssize_t label_stride,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        Py_ssize_t label,
        Py_ssize_t *label_count,
        size_t *seed,
    ) nogil:
        return samples[rand_int(0, n_samples, seed)]

cdef class TreeBuilder:

    cdef Py_ssize_t n_features
    cdef Py_ssize_t max_depth
    cdef Py_ssize_t min_samples_split
    cdef Py_ssize_t min_samples_leaf
    cdef double min_impurity_decrease

    cdef Py_ssize_t n_samples # no samples with non-zero weight
    cdef double n_weighted_samples

    cdef Py_ssize_t *samples
    cdef double *sample_weights

    cdef double *samples_branch_buffer
    cdef double *samples_branch
    cdef Py_ssize_t *pivot_buffer
    cdef Py_ssize_t *branch_count

    cdef Dataset dataset
    cdef CList distance_measures
    cdef readonly Tree tree
    cdef Criterion criterion
    cdef DistanceMeasureSampler distance_measure_sampler
    cdef PivotSampler pivot_sampler
    cdef size_t seed

    def __cinit__(
        self,
        np.ndarray x,
        np.ndarray sample_weights,
        PivotSampler pivot_sampler,
        DistanceMeasureSampler distance_measure_sampler,
        Criterion criterion,
        Tree tree,
        object random_state,
        Py_ssize_t n_features=1,
        Py_ssize_t max_depth=2**10,
        Py_ssize_t min_samples_split=2,
        Py_ssize_t min_samples_leaf=1,
        double min_impurity_decrease=0.0
    ):
        self.max_depth = max_depth
        self.n_features = n_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

        self.dataset = Dataset(x)
        self.criterion = criterion
        self.tree = tree
        self.distance_measure_sampler = distance_measure_sampler
        self.pivot_sampler = pivot_sampler
        self.distance_measures = tree.distance_measures
        self.seed = random_state.randint(0, RAND_R_MAX)

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
        self.samples_branch = <double*> malloc(sizeof(double) * self.dataset.n_samples)
        self.samples_branch_buffer = <double*> malloc(sizeof(double) * self.dataset.n_samples)

        self.pivot_buffer = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * self.criterion.n_labels)
        self.branch_count = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * self.criterion.n_labels)

        if (
            self.samples == NULL or
            self.samples_branch == NULL or
            self.samples_branch_buffer == NULL or
            self.pivot_buffer == NULL or
            self.branch_count == NULL
        ):
            raise MemoryError()

        cdef Py_ssize_t i, j
        j = 0
        self.n_weighted_samples = 0.0
        for i in range(self.dataset.n_samples):
            if sample_weights is None or sample_weights[i] != 0.0:
                self.samples[j] = i
                j += 1
                if sample_weights is not None:
                    self.n_weighted_samples += sample_weights[i]
                else:
                    self.n_weighted_samples += 1.0

        self.n_samples = j

        for i in range(self.distance_measures.size):
            (<DistanceMeasure>self.distance_measures.get(i)).reset(self.dataset, self.dataset)

    def __dealloc__(self):
        free(self.samples)
        free(self.samples_branch_buffer)
        free(self.pivot_buffer)
        free(self.samples_branch)
        free(self.branch_count)

    def build_tree(self):
        cdef Py_ssize_t max_depth = 0
        with nogil:
            self._build_tree(
                0,
                self.n_samples,
                0,
                -1,
                0,
                NAN,
                &max_depth,
            )
        self.tree._max_depth = max_depth

    cdef void _build_tree(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        Py_ssize_t depth,
        Py_ssize_t parent,
        Py_ssize_t branch,
        double impurity,
        Py_ssize_t *max_depth
    ) nogil:
        if depth > max_depth[0]:
            max_depth[0] = depth

        self.criterion.init(start, end, self.samples, self.sample_weights)
        cdef Py_ssize_t n_node_samples = end - start
        cdef bint is_leaf = (
            depth >= self.max_depth
            or n_node_samples < self.min_samples_split
            or n_node_samples < 2 * self.min_samples_leaf
        )

        cdef Py_ssize_t i
        if is_leaf:
            self._new_leaf_node(parent, branch)
            return

        if parent < 0:
            impurity = self.criterion.impurity()

        cdef Py_ssize_t j
        cdef Py_ssize_t node_id
        cdef Py_ssize_t current_split
        cdef Py_ssize_t split_start
        cdef double current_weight = 1.0
        cdef double p
        cdef Split split = self._split(start, end, impurity)
        is_leaf = (
            split.n_split < 1 or
            split.impurity_improvement <= self.min_impurity_decrease
        )

        if not is_leaf:
            node_id = self.tree.add_branch_node(
                parent,
                branch,
                n_node_samples,
                self.criterion.weighted_n_total,
                self.dataset,
                split.pivot,
                split.distance_measure,
                split.n_split,
            )
            split_start = start
            for current_split in range(split.n_split):
                self._build_tree(
                    split_start,
                    split.split_point[current_split],
                    depth + 1,
                    node_id,
                    current_split,
                    split.child_impurity[current_split],
                    max_depth,
                )
                split_start = split.split_point[current_split]

            self._build_tree(
                split.split_point[split.n_split - 1],
                end,
                depth + 1,
                node_id,
                split.n_split,
                split.child_impurity[split.n_split],
                max_depth,
            )
        else:
            self._new_leaf_node(parent, branch)

        free_split(&split)

    cdef void _new_leaf_node(self, Py_ssize_t parent, Py_ssize_t branch) nogil:
        cdef Py_ssize_t node_id = self.tree.add_leaf_node(parent, branch)
        cdef Py_ssize_t i
        for i in range(self.criterion.n_labels):
            self.tree.set_leaf_value(
                node_id,
                i,
                self.criterion.weighted_label_count[i] / self.criterion.weighted_n_total
            )

    cdef Split _split(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        double impurity_parent,
    ) nogil:
        cdef Py_ssize_t n_samples = end - start
        cdef Py_ssize_t n_branches = 0
        cdef Py_ssize_t i, j
        for i in range(self.criterion.n_labels):
            if self.criterion.label_count[i] > 0:
                n_branches += 1

        cdef Py_ssize_t label
        cdef Py_ssize_t pivot_index
        cdef Py_ssize_t best_distance_measure
        cdef double impurity
        cdef double best_impurity
        cdef Split split

        # Abort early if the samples of this branch has a single label
        if n_branches <= 1:
            split.n_split = 0
            split.impurity_improvement = INFINITY
            split.split_point = NULL
            split.pivot = NULL
            split.child_impurity = NULL
            split.distance_measure = -1
            return split
        else:
            split.pivot = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * self.criterion.n_labels)
            split.split_point = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * self.criterion.n_labels)
            split.child_impurity = <double*> malloc(sizeof(double) * self.criterion.n_labels)
            best_impurity = -INFINITY
            # TODO: we should not stop until a split with improvement is found.
            for _ in range(self.n_features):
                pivot_index = 0
                for label in range(self.criterion.n_labels):
                    if self.criterion.label_count[label] > 0:
                        split.pivot[pivot_index] = self.pivot_sampler.sample(
                            self.criterion.labels,
                            self.criterion.label_stride,
                            self.samples + start,
                            n_samples,
                            label,
                            self.criterion.label_count,
                            &self.seed,
                        )
                        pivot_index += 1

                split.distance_measure = self.distance_measure_sampler.sample(
                    self.samples + start, n_samples, &self.seed,
                )
                self._partition_pivots(
                    start, end, split.pivot, split.distance_measure, n_branches
                )
                self.criterion.reset(self.samples_branch)
                impurity = self.criterion.proxy_impurity(split.child_impurity, n_branches)
                if impurity > best_impurity:
                    best_impurity = impurity
                    best_distance_measure = split.distance_measure
                    memcpy(
                        self.pivot_buffer,
                        split.pivot,
                        sizeof(Py_ssize_t) * self.criterion.n_labels,
                    )
                    memcpy(
                        self.samples_branch_buffer + start,
                        self.samples_branch + start,
                        sizeof(double) * n_samples,
                    )

            # Restore the partitioning with the lowest impurity score
            memcpy(
                self.samples_branch + start,
                self.samples_branch_buffer + start,
                sizeof(double) * n_samples,
            )

            argsort(self.samples_branch + start, self.samples + start, n_samples)
            self.criterion.reset(self.samples_branch)
            self.criterion.child_impurity(split.child_impurity, n_branches)

            # Restore the pivots with the lowest impurity score
            memcpy(split.pivot, self.pivot_buffer, sizeof(Py_ssize_t) * self.criterion.n_labels)
            split.distance_measure = best_distance_measure

            # Shift unused pivots to the end. The number of used pivots might be lower
            # than the number of possible branches. We compute the number of splits
            # in the _find_split_points-method. The number of branches is the number
            # of splits + 1.
            memset(self.branch_count, 0, sizeof(Py_ssize_t) * self.criterion.n_labels)
            for i in range(start, end):
                self.branch_count[<Py_ssize_t> self.samples_branch[i]] += 1

            j = 0
            for i in range(n_branches):
                if self.branch_count[i] > 0:
                    split.pivot[j] = split.pivot[i]
                    split.child_impurity[j] = split.child_impurity[i]
                    j += 1
                else:
                    split.pivot[i] = -1
                    split.child_impurity[i] = -1

            self._find_split_points(start, end, &split)
            split.impurity_improvement = self.criterion.impurity_improvement(
                impurity_parent,
                split.child_impurity,
                split.n_split + 1,
                self.n_weighted_samples,
            )

            return split

    # Identify the indices where the values of the samples_branch array changes.
    cdef Py_ssize_t _find_split_points(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        Split *split
    ) nogil:
        cdef Py_ssize_t i, current_split
        current_split = 0
        for i in range(start + 1, end):
            if self.samples_branch[i] != self.samples_branch[i - 1]:
                split.split_point[current_split] = i
                current_split += 1

        # Set the number of splits to the number of actual split points.
        split.n_split = current_split

    # Partition the samples to the branch of the pivot with the smallest distance
    cdef void _partition_pivots(
        self,
        Py_ssize_t start,
        Py_ssize_t end,
        Py_ssize_t *pivots,
        Py_ssize_t distance_measure,
        Py_ssize_t n_branches
    ) nogil:
        cdef Py_ssize_t i, j
        cdef Py_ssize_t pivot
        cdef Py_ssize_t min_pivot
        cdef double min_dist = INFINITY
        cdef double dist
        for i in range(start, end):
            j = self.samples[i]
            min_dist = INFINITY
            for pivot in range(n_branches):
                dist = (
                    <DistanceMeasure>self.distance_measures.get(distance_measure)
                ).distance(self.dataset, pivots[pivot], self.dataset, j, 0)
                if dist < min_dist:
                    min_dist = dist
                    min_pivot = pivot
            self.samples_branch[i] = min_pivot
