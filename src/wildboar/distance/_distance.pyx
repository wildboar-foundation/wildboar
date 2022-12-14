# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np

from libc.math cimport NAN, sqrt
from libc.stdlib cimport free, malloc
from numpy cimport float64_t, intp_t, ndarray

from ..utils cimport _stats

from copy import deepcopy

from ..utils cimport TSArray

from ..utils._parallel import run_in_parallel
from ..utils.validation import check_array


cdef double EPSILON = 1e-13


cdef class MetricList(List):

    cdef int reset(self, Py_ssize_t metric, TSArray X, TSArray Y) nogil:
        return (<DistanceMeasure> self.get(metric)).reset(X, Y)

    cdef double distance(
        self, 
        Py_ssize_t metric, 
        TSArray X,
        Py_ssize_t x_index,
        TSArray Y,
        Py_ssize_t y_index,
        Py_ssize_t dim,
    ) nogil:
        return (<DistanceMeasure> self.get(metric)).distance(X, x_index, Y, y_index, dim)

    cdef double _distance(
        self,
        Py_ssize_t metric,
        const double *x,
        Py_ssize_t x_len,
        const double *y,
        Py_ssize_t y_len
    ) nogil:
        return (<DistanceMeasure> self.get(metric))._distance(x, x_len, y, y_len)


cdef class SubsequenceMetricList(List):

    cdef int reset(self, Py_ssize_t metric, TSArray X) nogil:
        return (<SubsequenceDistanceMeasure> self.get(metric)).reset(X)

    cdef int init_transient(
        self,
        Py_ssize_t metric,
        TSArray X,
        SubsequenceView *v,
        Py_ssize_t index,
        Py_ssize_t start,
        Py_ssize_t length,
        Py_ssize_t dim,
    ) nogil:
        return (<SubsequenceDistanceMeasure> self.get(metric)).init_transient(
            X, v, index, start, length, dim
        )

    cdef int init_persistent(
        self,
        Py_ssize_t metric, 
        TSArray X,
        SubsequenceView* v,
        Subsequence* s,
    ) nogil:
        return (<SubsequenceDistanceMeasure> self.get(metric)).init_persistent(X, v, s)

    cdef int free_transient(self, Py_ssize_t metric, SubsequenceView *t) nogil:
        return (<SubsequenceDistanceMeasure> self.get(metric)).free_transient(t)

    cdef int free_persistent(self, Py_ssize_t metric, Subsequence *t) nogil:
        return (<SubsequenceDistanceMeasure> self.get(metric)).free_persistent(t)

    cdef int from_array(
        self,
        Py_ssize_t metric, 
        Subsequence *s,
        object obj,
    ):
        return (<SubsequenceDistanceMeasure> self.get(metric)).from_array(s, obj)

    cdef object to_array(
        self, 
        Py_ssize_t metric, 
        Subsequence *s
    ):
        return (<SubsequenceDistanceMeasure> self.get(metric)).to_array(s)

    cdef double transient_distance(
        self,
        Py_ssize_t metric, 
        SubsequenceView *v,
        TSArray X,
        Py_ssize_t index,
        Py_ssize_t *return_index,
    ) nogil:
        return (<SubsequenceDistanceMeasure> self.get(metric)).transient_distance(
            v, X, index, return_index
        )


    cdef double persistent_distance(
        self,
        Py_ssize_t metric, 
        Subsequence *s,
        TSArray X,
        Py_ssize_t index,
        Py_ssize_t *return_index,
    ) nogil:
        return (<SubsequenceDistanceMeasure> self.get(metric)).persistent_distance(
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
    ) nogil:
        return (<SubsequenceDistanceMeasure> self.get(metric)).transient_matches(
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
    ) nogil:
        return (<SubsequenceDistanceMeasure> self.get(metric)).persistent_matches(
            s, X, index, threshold, distances, indices
        )


cdef int _ts_view_update_statistics(SubsequenceView *v, const double* sample) nogil:
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


cdef class SubsequenceDistanceMeasure:

    cdef int reset(self, TSArray X) nogil:
        pass

    cdef int init_transient(
        self,
        TSArray X,
        SubsequenceView *v,
        Py_ssize_t index,
        Py_ssize_t start,
        Py_ssize_t length,
        Py_ssize_t dim,
    ) nogil:
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
    ) nogil:
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

    cdef int free_transient(self, SubsequenceView *v) nogil:
        return 0

    cdef int free_persistent(self, Subsequence *v) nogil:
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
    ) nogil:
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
    ) nogil:
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
    ) nogil:
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
    ) nogil:
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
    ) nogil:
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
    ) nogil:
        return -1


cdef class ScaledSubsequenceDistanceMeasure(SubsequenceDistanceMeasure):

    cdef int init_transient(
        self,
        TSArray X,
        SubsequenceView *v,
        Py_ssize_t index,
        Py_ssize_t start,
        Py_ssize_t length,
        Py_ssize_t dim,
    ) nogil:
        cdef int err = SubsequenceDistanceMeasure.init_transient(
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
        cdef int err = SubsequenceDistanceMeasure.from_array(self, v, obj)
        if err == -1:
            return -1

        dim, arr = obj
        v.mean = np.mean(arr)
        v.std = np.std(arr)
        if v.std <= EPSILON:
            v.std = 0.0
        return 0


cdef class DistanceMeasure:

    cdef int reset(self, TSArray x, TSArray y) nogil:
        return 0

    cdef double distance(
        self,
        TSArray x,
        Py_ssize_t x_index,
        TSArray y,
        Py_ssize_t y_index,
        Py_ssize_t dim,
    ) nogil:
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
    ) nogil:
        return NAN

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
    cdef SubsequenceDistanceMeasure distance_measure

    def __cinit__(
        self, 
        double[:, :] distances,
        Py_ssize_t[:, :] min_indices,
        TSArray X, 
        SubsequenceDistanceMeasure distance_measure
    ):
        self.X = X
        self.distances = distances
        self.min_indices = min_indices
        self.distance_measure = distance_measure
        self.shapelets = NULL
        self.n_shapelets = 0

    def __dealloc__(self):
        if self.shapelets != NULL:
            for i in range(self.n_shapelets):
                self.distance_measure.free_persistent(self.shapelets[i])
                free(self.shapelets[i])
            free(self.shapelets)
            self.shapelets = NULL

    cdef void set_shapelets(self, list shapelets, Py_ssize_t dim):
        cdef Py_ssize_t i
        self.n_shapelets = len(shapelets)
        self.shapelets = <Subsequence**> malloc(sizeof(Subsequence*) * self. n_shapelets)
        cdef Subsequence *s
        for i in range(self.n_shapelets):
            s = <Subsequence*> malloc(sizeof(Subsequence))
            self.distance_measure.from_array(s, (dim, shapelets[i]))
            self.shapelets[i] = s

    @property
    def n_work(self):
        return self.X.shape[0]

    def __call__(self, Py_ssize_t job_id, Py_ssize_t offset, Py_ssize_t batch_size):
        cdef Py_ssize_t i, j, min_index
        cdef Subsequence *s
        cdef double distance
        cdef SubsequenceDistanceMeasure distance_measure = deepcopy(self.distance_measure)
        with nogil:
            distance_measure.reset(self.X)
            for i in range(offset, offset + batch_size):
                for j in range(self.n_shapelets):
                    s = self.shapelets[j]
                    distance = distance_measure.persistent_distance(
                        s, self.X, i, &min_index
                    )
                    self.distances[i, j] = distance
                    self.min_indices[i, j] = min_index


def _pairwise_subsequence_distance(
    list shapelets, 
    TSArray x, 
    int dim, 
    SubsequenceDistanceMeasure distance_measure,
    n_jobs,
):
    n_samples = x.shape[0]
    distances = np.empty((n_samples, len(shapelets)), dtype=np.double)
    min_indicies = np.empty((n_samples, len(shapelets)), dtype=np.intp)
    subsequence_distance = _PairwiseSubsequenceDistance(
        distances,
        min_indicies,
        x,
        distance_measure,
    )
    distance_measure.reset(x) # TODO: Move to _PairwiseSubsequenceDistance
    subsequence_distance.set_shapelets(shapelets, dim)
    run_in_parallel(subsequence_distance, n_jobs=n_jobs, require="sharedmem")
    return distances, min_indicies


def _paired_subsequence_distance(
    list shapelets,
    TSArray x,
    int dim,
    SubsequenceDistanceMeasure distance_measure
):
    cdef Py_ssize_t n_samples = x.shape[0]
    cdef Py_ssize_t i, min_index
    cdef double dist
    cdef Subsequence subsequence
    cdef double[:] distances = np.empty(n_samples, dtype=np.double)
    cdef Py_ssize_t[:] min_indices = np.empty(n_samples, dtype=np.intp)

    distance_measure.reset(x)
    for i in range(n_samples):
        distance_measure.from_array(&subsequence, (dim, shapelets[i]))
        with nogil:
            dist = distance_measure.persistent_distance(&subsequence, x, i, &min_index)
            distance_measure.free_persistent(&subsequence)
            distances[i] = dist
            min_indices[i] = min_index

    return distances.base, min_indices.base


def _subsequence_match(
    object y,
    TSArray x,
    double threshold,
    int dim,
    SubsequenceDistanceMeasure distance_measure,
    n_jobs,
):
    cdef Py_ssize_t n_samples = x.shape[0]
    cdef double *distances 
    cdef Py_ssize_t *indicies
    cdef Py_ssize_t i, n_matches

    cdef list distances_list = []
    cdef list indicies_list = []

    cdef Subsequence subsequence
    distance_measure.reset(x)
    distance_measure.from_array(&subsequence, (dim, y))
    for i in range(x.shape[0]):
        n_matches = distance_measure.persistent_matches(
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
    distance_measure.free_persistent(&subsequence)
    return indicies_list, distances_list


def _paired_subsequence_match(
    list y,
    TSArray x,
    double threshold,
    int dim,
    SubsequenceDistanceMeasure distance_measure,
    n_jobs,
):
    cdef Py_ssize_t n_samples = x.shape[0]
    cdef double *distances 
    cdef Py_ssize_t *indicies
    cdef Py_ssize_t i, n_matches

    cdef list distances_list = []
    cdef list indicies_list = []

    cdef Subsequence subsequence
    distance_measure.reset(x)
    for i in range(n_samples):
        distance_measure.from_array(&subsequence, (dim, y[i]))
        n_matches = distance_measure.persistent_matches(
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
        distance_measure.free_persistent(&subsequence)

    return indicies_list, distances_list


def _pairwise_distance(
    TSArray y,
    TSArray x,
    Py_ssize_t dim,
    DistanceMeasure distance_measure,
    n_jobs,
):
    cdef:
        Py_ssize_t y_samples = y.shape[0]
        Py_ssize_t x_samples = x.shape[0]

    cdef double[:, :] out = np.empty((y_samples, x_samples), dtype=float)
    cdef Py_ssize_t i, j
    cdef double dist

    with nogil:
        distance_measure.reset(y, x)
        for i in range(y_samples):
            for j in range(x_samples):
                dist = distance_measure.distance(y, i, x, j, dim)
                out[i, j] = dist

    return out.base


def _singleton_pairwise_distance(
    TSArray x, 
    Py_ssize_t dim, 
    DistanceMeasure distance_measure, 
    n_jobs
):
    cdef Py_ssize_t n_samples = x.shape[0]
    cdef double[:, :] out = np.zeros((n_samples, n_samples), dtype=float)
    cdef Py_ssize_t i, j
    cdef double dist

    with nogil:
        distance_measure.reset(x, x)
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = distance_measure.distance(x, i, x, j, dim)
                out[i, j] = dist
                out[j, i] = dist

    return out.base


def _paired_distance(
    TSArray y,
    TSArray x,
    Py_ssize_t dim,
    DistanceMeasure distance_measure,
    n_jobs,
):
    cdef Py_ssize_t n_samples = y.shape[0]
    cdef double[:] out = np.empty(n_samples, dtype=np.double)
    cdef Py_ssize_t i

    with nogil:
        distance_measure.reset(y, x)
        for i in range(n_samples):
            dist = distance_measure.distance(y, i, x, i, dim)
            out[i] = dist

    return out.base
