# cython: language_level=3
# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: initializedcheck=False

# Authors: Isak Samsten
# License: BSD 3 clause

from libc.math cimport NAN
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy
from numpy cimport uint32_t

from ..distance._distance cimport MetricList
from ..utils cimport TSArray
from ..utils._misc cimport to_ndarray_double
from ..utils._rand cimport RAND_R_MAX, RandomSampler, rand_int

from ..distance import _METRICS

from ._feature cimport Feature, FeatureEngineer


cdef struct TransientPivot:
    Py_ssize_t sample
    Py_ssize_t metric

cdef struct PersitentPivot:
    double *data
    Py_ssize_t length
    Py_ssize_t metric

cdef class PivotFeatureEngineer(FeatureEngineer):
    cdef Py_ssize_t n_pivots
    cdef RandomSampler sampler
    cdef MetricList metrics

    def __cinit__(self, Py_ssize_t n_pivots, list metrics, RandomSampler sampler):
        self.n_pivots = n_pivots
        self.metrics = MetricList(metrics)
        self.sampler = sampler
        

    def __reduce__(self):
        return self.__class__, (self.n_pivots, self.metrics.py_list, self.sampler)

    cdef int reset(self, TSArray X) noexcept nogil:
        cdef Py_ssize_t i
        for i in range(self.metrics.size):
            self.metrics.reset(i, X, X)
        return 0

    cdef Py_ssize_t get_n_features(self, TSArray X) noexcept nogil:
        return self.n_pivots

    cdef Py_ssize_t get_n_outputs(self, TSArray X) noexcept nogil:
        return self.get_n_features(X)

    cdef Py_ssize_t next_feature(
            self,
            Py_ssize_t feature_id,
            TSArray X,
            Py_ssize_t *samples,
            Py_ssize_t n_samples,
            Feature *transient,
            uint32_t *seed,
    ) noexcept nogil:
        cdef TransientPivot *pivot = <TransientPivot*> malloc(sizeof(TransientPivot))
        pivot.sample = samples[rand_int(0, n_samples, seed)]
        pivot.metric = self.sampler.rand_int(seed)
        transient.dim = rand_int(0, X.shape[1], seed)
        transient.feature = pivot
        return 0

    cdef Py_ssize_t free_transient_feature(self, Feature *feature) noexcept nogil:
        if feature.feature != NULL:
            free(feature.feature)
            feature.feature = NULL
        return 0

    cdef Py_ssize_t free_persistent_feature(self, Feature *feature) noexcept nogil:
        cdef PersitentPivot *pivot
        if feature.feature != NULL:
            pivot = <PersitentPivot*> feature.feature
            free(pivot.data)
            free(feature.feature)
            feature.feature = NULL
        return 0

    cdef Py_ssize_t init_persistent_feature(
            self,
            TSArray X,
            Feature *transient,
            Feature *persistent
    ) noexcept nogil:
        cdef TransientPivot *pivot = <TransientPivot*> transient.feature
        cdef PersitentPivot *persistent_pivot = <PersitentPivot*> malloc(sizeof(PersitentPivot))
        
        persistent_pivot.data = <double*> malloc(sizeof(double) * X.shape[2])
        persistent_pivot.metric = pivot.metric
        persistent_pivot.length = X.shape[2]
        memcpy(
            persistent_pivot.data, 
            &X[pivot.sample, transient.dim, 0], 
            sizeof(double) * X.shape[2]
        )
        
        persistent.dim = transient.dim
        persistent.feature = persistent_pivot
        return 0

    cdef double transient_feature_value(
            self,
            Feature *feature,
            TSArray X,
            Py_ssize_t sample
    ) noexcept nogil:
        cdef TransientPivot* pivot = <TransientPivot*>feature.feature
        return self.metrics.distance(
            pivot.metric, X, sample, X, pivot.sample, feature.dim
        )

    cdef double persistent_feature_value(
            self,
            Feature *feature,
            TSArray X,
            Py_ssize_t sample
    ) noexcept nogil:
        cdef PersitentPivot* pivot = <PersitentPivot*> feature.feature
        return self.metrics._distance(
            pivot.metric,
            &X[sample, feature.dim, 0],
            X.shape[2],
            pivot.data,
            X.shape[2],
        )

    cdef Py_ssize_t transient_feature_fill(
            self,
            Feature *feature,
            TSArray X,
            Py_ssize_t sample,
            double[:, :] out,
            Py_ssize_t out_sample,
            Py_ssize_t out_feature,
    ) noexcept nogil:
        out[out_sample, out_feature] = self.transient_feature_value(feature, X, sample)
        return 0

    cdef Py_ssize_t persistent_feature_fill(
            self,
            Feature *feature,
            TSArray X,
            Py_ssize_t sample,
            double[:, :] out,
            Py_ssize_t out_sample,
            Py_ssize_t out_feature,
    ) noexcept nogil:
        out[out_sample, out_feature] = self.persistent_feature_value(feature, X, sample)
        return 0

    cdef object persistent_feature_to_object(self, Feature *feature):
        cdef PersitentPivot *pivot = <PersitentPivot*> feature.feature
        return feature.dim, (pivot.metric, to_ndarray_double(pivot.data, pivot.length))

    cdef Py_ssize_t persistent_feature_from_object(self, object object, Feature *feature):
        dim, (metric, arr) = object
        cdef PersitentPivot *pivot = <PersitentPivot*> malloc(sizeof(PersitentPivot))
        cdef double *data = <double*> malloc(sizeof(double) * arr.size)
        cdef Py_ssize_t i
        for i in range(arr.size):
            data[i] = arr[i]
        pivot.data = data
        pivot.length = arr.size
        pivot.metric = metric

        feature.dim = dim
        feature.feature = pivot
        return 0
