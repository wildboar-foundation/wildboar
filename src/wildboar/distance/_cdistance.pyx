# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: initializedcheck=False

# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np

from libc.math cimport NAN, sqrt, INFINITY
from libc.stdlib cimport free, malloc
from numpy cimport float64_t, intp_t, ndarray

from ..utils cimport _stats
from ..utils._misc cimport Heap

from copy import deepcopy

from ..utils cimport TSArray
from ..utils._misc cimport List

from ..utils._parallel import run_in_parallel
from ..utils.validation import check_array


cdef class MetricList(List):

    cdef int reset(self, Py_ssize_t metric, TSArray X, TSArray Y) noexcept nogil:
        return (<Metric> self.get(metric)).reset(X, Y)

    cdef double distance(
        self,
        Py_ssize_t metric,
        TSArray X,
        Py_ssize_t x_index,
        TSArray Y,
        Py_ssize_t y_index,
        Py_ssize_t dim,
    ) noexcept nogil:
        return (<Metric> self.get(metric)).distance(X, x_index, Y, y_index, dim)

    cdef double _distance(
        self,
        Py_ssize_t metric,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len
    ) noexcept nogil:
        return (<Metric> self.get(metric))._distance(x, x_len, y, y_len)


cdef class SubsequenceMetricList(List):

    cdef int reset(self, Py_ssize_t metric, TSArray X) noexcept nogil:
        return (<SubsequenceMetric> self.get(metric)).reset(X)

    cdef int init_transient(
        self,
        Py_ssize_t metric,
        TSArray X,
        SubsequenceView *v,
        Py_ssize_t index,
        Py_ssize_t start,
        Py_ssize_t length,
        Py_ssize_t dim,
    ) noexcept nogil:
        return (<SubsequenceMetric> self.get(metric)).init_transient(
            X, v, index, start, length, dim
        )

    cdef int init_persistent(
        self,
        Py_ssize_t metric,
        TSArray X,
        SubsequenceView* v,
        Subsequence* s,
    ) noexcept nogil:
        return (<SubsequenceMetric> self.get(metric)).init_persistent(X, v, s)

    cdef int free_transient(self, Py_ssize_t metric, SubsequenceView *t) noexcept nogil:
        return (<SubsequenceMetric> self.get(metric)).free_transient(t)

    cdef int free_persistent(self, Py_ssize_t metric, Subsequence *t) noexcept nogil:
        return (<SubsequenceMetric> self.get(metric)).free_persistent(t)

    cdef int from_array(
        self,
        Py_ssize_t metric,
        Subsequence *s,
        object obj,
    ):
        return (<SubsequenceMetric> self.get(metric)).from_array(s, obj)

    cdef object to_array(
        self,
        Py_ssize_t metric,
        Subsequence *s
    ):
        return (<SubsequenceMetric> self.get(metric)).to_array(s)

    cdef double transient_distance(
        self,
        Py_ssize_t metric,
        SubsequenceView *v,
        TSArray X,
        Py_ssize_t index,
        Py_ssize_t *return_index,
    ) noexcept nogil:
        return (<SubsequenceMetric> self.get(metric)).transient_distance(
            v, X, index, return_index
        )

    cdef double persistent_distance(
        self,
        Py_ssize_t metric,
        Subsequence *s,
        TSArray X,
        Py_ssize_t index,
        Py_ssize_t *return_index,
    ) noexcept nogil:
        return (<SubsequenceMetric> self.get(metric)).persistent_distance(
            s, X, index, return_index
        )

    cdef Py_ssize_t transient_matches(
        self,
        Py_ssize_t metric,
        SubsequenceView *v,
        TSArray X,
        Py_ssize_t index,
        double threshold,
        double **distances,
        Py_ssize_t **indices,
    ) noexcept nogil:
        return (<SubsequenceMetric> self.get(metric)).transient_matches(
            v, X, index, threshold, distances, indices
        )

    cdef Py_ssize_t persistent_matches(
        self,
        Py_ssize_t metric,
        Subsequence *s,
        TSArray X,
        Py_ssize_t index,
        double threshold,
        double **distances,
        Py_ssize_t **indices,
    ) noexcept nogil:
        return (<SubsequenceMetric> self.get(metric)).persistent_matches(
            s, X, index, threshold, distances, indices
        )


cdef int _ts_view_update_statistics(
    SubsequenceView *v, const double* sample
) noexcept nogil:
    """Update the mean and standard deviation of a shapelet info struct """
    cdef double ex = 0
    cdef double ex2 = 0
    cdef Py_ssize_t i
    for i in range(v.length):
        current_value = sample[i]
        ex += current_value
        ex2 += current_value ** 2

    v.mean = ex / v.length
    ex2 = ex2 / v.length - v.mean * v.mean
    if ex2 > EPSILON:
        v.std = sqrt(ex2)
    else:
        v.std = 0.0
    return 0


cdef class SubsequenceMetric:

    cdef int reset(self, TSArray X) noexcept nogil:
        pass

    cdef int init_transient(
        self,
        TSArray X,
        SubsequenceView *v,
        Py_ssize_t index,
        Py_ssize_t start,
        Py_ssize_t length,
        Py_ssize_t dim,
    ) noexcept nogil:
        v.index = index
        v.dim = dim
        v.start = start
        v.length = length
        v.mean = NAN
        v.std = NAN
        v.extra = NULL
        return 0

    cdef int init_persistent(
        self, TSArray X, SubsequenceView* v, Subsequence* s
    ) noexcept nogil:
        s.dim = v.dim
        s.length = v.length
        s.mean = v.mean
        s.std = v.std
        s.ts_start = v.start
        s.ts_index = v.index
        s.extra = NULL
        s.data = <double*> malloc(sizeof(double) * v.length)
        if s.data == NULL:
            return -1

        cdef const double *sample = &X[v.index, v.dim, v.start]
        cdef Py_ssize_t i
        for i in range(v.length):
            s.data[i] = sample[i]
        return 0

    cdef int free_transient(self, SubsequenceView *v) noexcept nogil:
        return 0

    cdef int free_persistent(self, Subsequence *v) noexcept nogil:
        if v.data != NULL:
            free(v.data)
            v.data = NULL
        return 0

    cdef int from_array(self, Subsequence *v, object obj):
        dim, arr = obj
        v.dim = dim
        v.length = arr.shape[0]
        v.mean = NAN
        v.std = NAN
        v.data = <double*> malloc(v.length * sizeof(double))
        v.extra = NULL
        if v.data == NULL:
            return -1

        cdef Py_ssize_t i
        for i in range(v.length):
            v.data[i] = arr[i]
        return 0

    cdef object to_array(self, Subsequence *s):
        cdef Py_ssize_t j
        cdef ndarray[float64_t] arr = np.empty(s.length, dtype=float)
        with nogil:
            for j in range(s.length):
                arr[j] = s.data[j]

        return (s.dim, arr)

    cdef double transient_distance(
        self,
        SubsequenceView *v,
        TSArray X,
        Py_ssize_t index,
        Py_ssize_t *return_index=NULL,
    ) noexcept nogil:
        return self._distance(
            &X[v.index, v.dim, v.start],
            v.length,
            v.mean,
            v.std,
            v.extra,
            &X[index, v.dim, 0],
            X.shape[2],
            return_index,
        )

    cdef double persistent_distance(
        self,
        Subsequence *v,
        TSArray X,
        Py_ssize_t index,
        Py_ssize_t *return_index=NULL,
    ) noexcept nogil:
        return self._distance(
            v.data,
            v.length,
            v.mean,
            v.std,
            v.extra,
            &X[index, v.dim, 0],
            X.shape[2],
            return_index,
        )

    cdef Py_ssize_t transient_matches(
        self,
        SubsequenceView *v,
        TSArray X,
        Py_ssize_t index,
        double threshold,
        double **distances,
        Py_ssize_t **indicies,
    ) noexcept nogil:
        return self._matches(
            &X[v.index, v.dim, v.start],
            v.length,
            v.mean,
            v.std,
            v.extra,
            &X[index, v.dim, 0],
            X.shape[2],
            threshold,
            distances,
            indicies,
        )

    cdef Py_ssize_t persistent_matches(
        self,
        Subsequence *s,
        TSArray X,
        Py_ssize_t index,
        double threshold,
        double **distances,
        Py_ssize_t **indicies,
    ) noexcept nogil:
        return self._matches(
            s.data,
            s.length,
            s.mean,
            s.std,
            s.extra,
            &X[index, s.dim, 0],
            X.shape[2],
            threshold,
            distances,
            indicies,
        )

    cdef double _distance(
        self,
        const double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        const double *x,
        Py_ssize_t x_len,
        Py_ssize_t *return_index=NULL,
    ) noexcept nogil:
        return NAN

    cdef Py_ssize_t _matches(
        self,
        const double *s,
        Py_ssize_t s_len,
        double s_mean,
        double s_std,
        void *s_extra,
        const double *x,
        Py_ssize_t x_len,
        double threshold,
        double **distances,
        Py_ssize_t **indicies,
    ) noexcept nogil:
        return -1


cdef class ScaledSubsequenceMetric(SubsequenceMetric):

    cdef int init_transient(
        self,
        TSArray X,
        SubsequenceView *v,
        Py_ssize_t index,
        Py_ssize_t start,
        Py_ssize_t length,
        Py_ssize_t dim,
    ) noexcept nogil:
        cdef int err = SubsequenceMetric.init_transient(
            self, X, v, index, start, length, dim
        )
        if err < 0:
            return err
        _ts_view_update_statistics(v, &X[v.index, v.dim, v.start])
        return 0

    cdef int from_array(
        self,
        Subsequence *v,
        object obj,
    ):
        cdef int err = SubsequenceMetric.from_array(self, v, obj)
        if err == -1:
            return -1

        dim, arr = obj
        v.mean = np.mean(arr)
        v.std = np.std(arr)
        if v.std <= EPSILON:
            v.std = 0.0
        return 0


cdef class Metric:

    cdef int reset(self, TSArray x, TSArray y) noexcept nogil:
        return 0

    cdef double distance(
        self,
        TSArray x,
        Py_ssize_t x_index,
        TSArray y,
        Py_ssize_t y_index,
        Py_ssize_t dim,
    ) noexcept nogil:
        return self._distance(
            &x[x_index, dim, 0],
            x.shape[2],
            &y[y_index, dim, 0],
            y.shape[2],
        )

    cdef double _distance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len
    ) noexcept nogil:
        return NAN

    # Default implementation. Delegates to _lbdistance.
    cdef MetricState lbdistance(
        self,
        TSArray x,
        Py_ssize_t x_index,
        TSArray y,
        Py_ssize_t y_index,
        Py_ssize_t dim,
        double *distance,
    ) noexcept nogil:
        return self._lbdistance(
           &x[x_index, dim, 0],
           x.shape[2],
           &y[y_index, dim, 0],
           y.shape[2],
           distance,
        )

    # Default implementation. Delegates to _distance,
    # without lower bounding.
    cdef MetricState _lbdistance(
        self,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len,
        double *distance,
    ) noexcept nogil:
        distance[0] = self._distance(x, x_len, y, y_len)
        return MetricState.VALID

    @property
    def is_elastic(self):
        return False


cdef ndarray[intp_t] _new_match_array(Py_ssize_t *matches, Py_ssize_t n_matches):
    if n_matches > 0:
        match_array = np.empty(n_matches, dtype=np.intp)
        for i in range(n_matches):
            match_array[i] = matches[i]
        return match_array
    else:
        return None


cdef ndarray[intp_t] _new_distance_array(double *distances, Py_ssize_t n_matches):
    if n_matches > 0:
        dist_array = np.empty(n_matches, dtype=np.double)
        for i in range(n_matches):
            dist_array[i] = distances[i]
        return dist_array
    else:
        return None


cdef class _PairwiseSubsequenceDistance:

    cdef TSArray X
    cdef Py_ssize_t[:, :] min_indices
    cdef double[:, :] distances,
    cdef Subsequence **shapelets
    cdef Py_ssize_t n_shapelets
    cdef SubsequenceMetric metric

    def __cinit__(
        self,
        double[:, :] distances,
        Py_ssize_t[:, :] min_indices,
        TSArray X,
        SubsequenceMetric metric
    ):
        self.X = X
        self.distances = distances
        self.min_indices = min_indices
        self.metric = metric
        self.shapelets = NULL
        self.n_shapelets = 0

    def __dealloc__(self):
        if self.shapelets != NULL:
            for i in range(self.n_shapelets):
                self.metric.free_persistent(self.shapelets[i])
                free(self.shapelets[i])
            free(self.shapelets)
            self.shapelets = NULL

    cdef void set_shapelets(self, list shapelets, Py_ssize_t dim):
        cdef Py_ssize_t i
        self.n_shapelets = len(shapelets)
        self.shapelets = <Subsequence**> malloc(sizeof(Subsequence*) * self.n_shapelets)
        cdef Subsequence *s
        for i in range(self.n_shapelets):
            s = <Subsequence*> malloc(sizeof(Subsequence))
            self.metric.from_array(s, (dim, shapelets[i]))
            self.shapelets[i] = s

    @property
    def n_work(self):
        return self.X.shape[0]

    def __call__(self, Py_ssize_t job_id, Py_ssize_t offset, Py_ssize_t batch_size):
        cdef Py_ssize_t i, j, min_index
        cdef Subsequence *s
        cdef double distance
        cdef SubsequenceMetric metric = deepcopy(self.metric)
        with nogil:
            metric.reset(self.X)
            for i in range(offset, offset + batch_size):
                for j in range(self.n_shapelets):
                    s = self.shapelets[j]
                    distance = metric.persistent_distance(
                        s, self.X, i, &min_index
                    )
                    self.distances[i, j] = distance
                    self.min_indices[i, j] = min_index


def _pairwise_subsequence_distance(
    list shapelets,
    TSArray x,
    int dim,
    SubsequenceMetric metric,
    n_jobs,
):
    n_samples = x.shape[0]
    distances = np.empty((n_samples, len(shapelets)), dtype=np.double)
    min_indicies = np.empty((n_samples, len(shapelets)), dtype=np.intp)
    subsequence_distance = _PairwiseSubsequenceDistance(
        distances,
        min_indicies,
        x,
        metric,
    )
    metric.reset(x)  # TODO: Move to _PairwiseSubsequenceDistance
    subsequence_distance.set_shapelets(shapelets, dim)
    run_in_parallel(subsequence_distance, n_jobs=n_jobs, require="sharedmem")
    return distances, min_indicies


def _paired_subsequence_distance(
    list shapelets,
    TSArray x,
    int dim,
    SubsequenceMetric metric
):
    cdef Py_ssize_t n_samples = x.shape[0]
    cdef Py_ssize_t i, min_index
    cdef double dist
    cdef Subsequence subsequence
    cdef double[:] distances = np.empty(n_samples, dtype=np.double)
    cdef Py_ssize_t[:] min_indices = np.empty(n_samples, dtype=np.intp)

    metric.reset(x)
    for i in range(n_samples):
        metric.from_array(&subsequence, (dim, shapelets[i]))
        with nogil:
            dist = metric.persistent_distance(&subsequence, x, i, &min_index)
            metric.free_persistent(&subsequence)
            distances[i] = dist
            min_indices[i] = min_index

    return distances.base, min_indices.base


def _subsequence_match(
    object y,
    TSArray x,
    double threshold,
    int dim,
    SubsequenceMetric metric,
    n_jobs,
):
    cdef Py_ssize_t n_samples = x.shape[0]
    cdef double *distances
    cdef Py_ssize_t *indicies
    cdef Py_ssize_t i, n_matches

    cdef list distances_list = []
    cdef list indicies_list = []

    cdef Subsequence subsequence
    metric.reset(x)
    metric.from_array(&subsequence, (dim, y))
    for i in range(x.shape[0]):
        n_matches = metric.persistent_matches(
            &subsequence,
            x,
            i,
            threshold,
            &distances,
            &indicies,
        )
        indicies_list.append(_new_match_array(indicies, n_matches))
        distances_list.append(_new_distance_array(distances, n_matches))
        free(distances)
        free(indicies)
    metric.free_persistent(&subsequence)
    return indicies_list, distances_list


def _paired_subsequence_match(
    list y,
    TSArray x,
    double threshold,
    int dim,
    SubsequenceMetric metric,
    n_jobs,
):
    cdef Py_ssize_t n_samples = x.shape[0]
    cdef double *distances
    cdef Py_ssize_t *indicies
    cdef Py_ssize_t i, n_matches

    cdef list distances_list = []
    cdef list indicies_list = []

    cdef Subsequence subsequence
    metric.reset(x)
    for i in range(n_samples):
        metric.from_array(&subsequence, (dim, y[i]))
        n_matches = metric.persistent_matches(
            &subsequence,
            x,
            i,
            threshold,
            &distances,
            &indicies,
        )
        indicies_list.append(_new_match_array(indicies, n_matches))
        distances_list.append(_new_distance_array(distances, n_matches))
        free(distances)
        free(indicies)
        metric.free_persistent(&subsequence)

    return indicies_list, distances_list


cdef class _PairwiseDistanceBatch:
    cdef TSArray x
    cdef TSArray y
    cdef Py_ssize_t dim
    cdef Metric metric

    cdef double[:, :] distances

    def __init__(
        self,
        TSArray x,
        TSArray y,
        Py_ssize_t dim,
        Metric metric,
        double[:, :] distances
    ):
        self.x = x
        self.y = y
        self.dim = dim
        self.metric = metric
        self.distances = distances

    @property
    def n_work(self):
        return self.x.shape[0]

    def __call__(self, Py_ssize_t job_id, Py_ssize_t offset, Py_ssize_t batch_size):
        cdef Metric metric = deepcopy(self.metric)
        cdef Py_ssize_t i, j
        cdef Py_ssize_t y_samples = self.y.shape[0]

        with nogil:
            metric.reset(self.x, self.y)
            for i in range(offset, offset + batch_size):
                for j in range(y_samples):
                    self.distances[i, j] = metric.distance(
                        self.x, i, self.y, j, self.dim
                    )


def _pairwise_distance(
    TSArray x,
    TSArray y,
    Py_ssize_t dim,
    Metric metric,
    n_jobs,
):

    cdef double[:, :] out = np.empty((x.shape[0], y.shape[0]), dtype=float)
    run_in_parallel(
        _PairwiseDistanceBatch(
            x,
            y,
            dim,
            metric,
            out,
        ),
        n_jobs=n_jobs,
        require="sharedmem",
    )

    return out.base


cdef class _SingletonPairwiseDistanceBatch:
    cdef TSArray x
    cdef Py_ssize_t dim
    cdef Metric metric

    cdef double[:, :] distances

    def __init__(
        self,
        TSArray x,
        Py_ssize_t dim,
        Metric metric,
        double[:, :] distances
    ):
        self.x = x
        self.dim = dim
        self.metric = metric
        self.distances = distances

    @property
    def n_work(self):
        return self.x.shape[0]

    def __call__(self, Py_ssize_t job_id, Py_ssize_t offset, Py_ssize_t batch_size):
        cdef Metric metric = deepcopy(self.metric)
        cdef Py_ssize_t i, j
        cdef Py_ssize_t n_samples = self.x.shape[0]
        cdef double dist

        with nogil:
            metric.reset(self.x, self.x)
            for i in range(offset, offset + batch_size):
                for j in range(i + 1, n_samples):
                    dist = metric.distance(
                        self.x, i, self.x, j, self.dim
                    )
                    self.distances[i, j] = dist
                    self.distances[j, i] = dist


def _singleton_pairwise_distance(
    TSArray x,
    Py_ssize_t dim,
    Metric metric,
    n_jobs
):
    cdef Py_ssize_t n_samples = x.shape[0]
    cdef double[:, :] out = np.zeros((n_samples, n_samples), dtype=float)

    run_in_parallel(
        _SingletonPairwiseDistanceBatch(
            x,
            dim,
            metric,
            out,
        ),
        n_jobs=n_jobs,
        require="sharedmem",
    )
    return out.base


cdef class _ArgMinBatch:
    cdef double[:, :] distances
    cdef Py_ssize_t[:, :] indices

    cdef TSArray x
    cdef TSArray y
    cdef Metric metric
    cdef Py_ssize_t dim

    def __init__(
        self,
        TSArray x,
        TSArray y,
        Metric metric,
        Py_ssize_t dim,
        double[:, :] distances,
        Py_ssize_t[:, :] indices
    ):
        self.x = x
        self.y = y
        self.metric = metric
        self.dim = dim
        self.distances = distances
        self.indices = indices

    @property
    def n_work(self):
        return self.x.shape[0]

    def __call__(self, Py_ssize_t job_id, Py_ssize_t offset, Py_ssize_t batch_size):
        cdef MetricState state
        cdef double distance
        cdef Py_ssize_t i, j

        cdef Py_ssize_t k = self.distances.shape[1]
        cdef Py_ssize_t y_samples = self.y.shape[0]
        cdef Heap heap = Heap(k)
        cdef Metric metric = deepcopy(self.metric)

        with nogil:
            metric.reset(self.x, self.y)
            for i in range(offset, offset + batch_size):
                distance = INFINITY
                heap.reset()

                for j in range(y_samples):
                    state = metric.lbdistance(self.x, i, self.y, j, self.dim, &distance)
                    if state == MetricState.VALID:
                        heap.push(j, distance)

                        if heap.isfull():
                            distance = heap.max().value
                        else:
                            distance = INFINITY

                for j in range(k):
                    self.indices[i, j] = heap.get(j).index
                    self.distances[i, j] = heap.get(j).value


def _argmin_distance(
    TSArray x,
    TSArray y,
    Py_ssize_t dim,
    Metric metric,
    Py_ssize_t k,
    n_jobs
):

    cdef Py_ssize_t x_samples = x.shape[0]
    cdef Py_ssize_t y_samples = y.shape[0]

    cdef Py_ssize_t[:, :] indices = np.zeros((x_samples, k), dtype=int)
    cdef double[:, :] values = np.zeros((x_samples, k), dtype=float)

    run_in_parallel(
        _ArgMinBatch(
            x,
            y,
            metric,
            dim,
            values,
            indices,
        ),
        n_jobs=n_jobs,
        require="sharedmem"
    )

    return indices.base, values.base


cdef class _PairedDistanceBatch:
    cdef TSArray x
    cdef TSArray y
    cdef Py_ssize_t dim
    cdef Metric metric

    cdef double[:] distances

    def __init__(
        self,
        TSArray x,
        TSArray y,
        Py_ssize_t dim,
        Metric metric,
        double[:] distances
    ):
        self.x = x
        self.y = y
        self.dim = dim
        self.metric = metric
        self.distances = distances

    @property
    def n_work(self):
        return self.x.shape[0]

    def __call__(self, Py_ssize_t job_id, Py_ssize_t offset, Py_ssize_t batch_size):
        cdef Metric metric = deepcopy(self.metric)
        cdef Py_ssize_t i
        with nogil:
            metric.reset(self.x, self.y)
            for i in range(offset, offset + batch_size):
                self.distances[i] = metric.distance(self.x, i, self.y, i, self.dim)


def _paired_distance(
    TSArray y,
    TSArray x,
    Py_ssize_t dim,
    Metric metric,
    n_jobs,
):
    cdef double[:] out = np.empty(y.shape[0], dtype=float)
    run_in_parallel(
        _PairedDistanceBatch(
            x,
            y,
            dim,
            metric,
            out,
        ),
        n_jobs=n_jobs,
        require="sharedmem",
    )

    return out.base
