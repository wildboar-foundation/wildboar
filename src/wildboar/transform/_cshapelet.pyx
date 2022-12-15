# cython: language_level=3
# cython: boundscheck=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Isak Samsten
# License: BSD 3 clause
import numpy as np

from libc.stdlib cimport free, malloc
from numpy cimport uint32_t

from ..distance._distance cimport (
    Subsequence,
    SubsequenceMetric,
    SubsequenceMetricList,
    SubsequenceView,
)
from ..utils cimport TSArray
from ..utils._rand cimport (
    RAND_R_MAX,
    VoseRand,
    rand_int,
    vose_rand_free,
    vose_rand_init,
    vose_rand_int,
    vose_rand_precompute,
)
from ._feature cimport Feature, FeatureEngineer


cdef class ShapeletFeatureEngineer(FeatureEngineer):

    cdef Py_ssize_t min_shapelet_size
    cdef Py_ssize_t max_shapelet_size

    cdef readonly SubsequenceMetric metric

    def __init__(self, metric, min_shapelet_size, max_shapelet_size):
        self.metric = metric
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size

    cdef int reset(self, TSArray X) nogil:
        self.metric.reset(X)
        return 1

    cdef Py_ssize_t free_transient_feature(self, Feature *feature) nogil:
        if feature.feature != NULL:
            self.metric.free_transient(<SubsequenceView*> feature.feature)
            free(feature.feature)

    cdef Py_ssize_t free_persistent_feature(self, Feature *feature) nogil:
        cdef Subsequence *s
        if feature.feature != NULL:
            self.metric.free_persistent(<Subsequence*> feature.feature)
            free(feature.feature)

    cdef Py_ssize_t init_persistent_feature(
        self, 
        TSArray X,
        Feature *transient, 
        Feature *persistent
    ) nogil:
        cdef SubsequenceView *v = <SubsequenceView*> transient.feature
        cdef Subsequence *s = <Subsequence*> malloc(sizeof(Subsequence))
        self.metric.init_persistent(X, v, s)
        persistent.dim = transient.dim
        persistent.feature = s
        return 1

    cdef double transient_feature_value(
        self,
        Feature *feature,
        TSArray X,
        Py_ssize_t sample
    ) nogil:
        return self.metric.transient_distance(
            <SubsequenceView*> feature.feature, X, sample
        )

    cdef double persistent_feature_value(
        self,
        Feature *feature,
        TSArray X,
        Py_ssize_t sample
    ) nogil:
        return self.metric.persistent_distance(
            <Subsequence*> feature.feature, X, sample
        )

    cdef Py_ssize_t transient_feature_fill(
        self, 
        Feature *feature, 
        TSArray X, 
        Py_ssize_t sample,
        double[:, :] out,
        Py_ssize_t out_sample,
        Py_ssize_t feature_id,
    ) nogil:
        out[out_sample, feature_id] = self.metric.transient_distance(
            <SubsequenceView*> feature.feature, X, sample
        )
        return 0

    cdef Py_ssize_t persistent_feature_fill(
        self, 
        Feature *feature, 
        TSArray X, 
        Py_ssize_t sample,
        double[:, :] out,
        Py_ssize_t out_sample,
        Py_ssize_t feature_id,
    ) nogil:
        out[out_sample, feature_id] = self.metric.persistent_distance(
            <Subsequence*> feature.feature, X, sample
        )
        return 0

    cdef object persistent_feature_to_object(self, Feature *feature):
        return feature.dim, self.metric.to_array(<Subsequence*>feature.feature)

    cdef Py_ssize_t persistent_feature_from_object(self, object object, Feature *feature):
        cdef Subsequence *s = <Subsequence*> malloc(sizeof(Subsequence))
        dim, obj = object
        self.metric.from_array(s, obj)
        feature.dim = dim
        feature.feature = s
        return 0

cdef class RandomShapeletFeatureEngineer(ShapeletFeatureEngineer):

    cdef Py_ssize_t n_shapelets

    def __init__(
        self, metric, min_shapelet_size, max_shapelet_size, n_shapelets
    ):
        super().__init__(metric, min_shapelet_size, max_shapelet_size)
        self.n_shapelets = n_shapelets

    cdef Py_ssize_t get_n_features(self, TSArray X) nogil:
        return self.n_shapelets

    cdef Py_ssize_t next_feature(
        self,
        Py_ssize_t feature_id,
        TSArray X, 
        Py_ssize_t *samples, 
        Py_ssize_t n_samples,
        Feature *transient,
        uint32_t *random_seed
    ) nogil:
        if feature_id >= self.n_shapelets:
            return -1
        
        cdef Py_ssize_t shapelet_length
        cdef Py_ssize_t shapelet_start
        cdef Py_ssize_t shapelet_index
        cdef Py_ssize_t shapelet_dim
        cdef SubsequenceView *v = <SubsequenceView*> malloc(sizeof(SubsequenceView))

        shapelet_length = rand_int(
            self.min_shapelet_size, self.max_shapelet_size, random_seed)
        shapelet_start = rand_int(
            0, X.shape[2] - shapelet_length, random_seed)
        shapelet_index = samples[rand_int(0, n_samples, random_seed)]
        if X.shape[1] > 1:
            shapelet_dim = rand_int(0, X.shape[1], random_seed)
        else:
            shapelet_dim = 0

        transient.dim = shapelet_dim
        self.metric.init_transient(
            X,
            v,
            shapelet_index,
            shapelet_start,
            shapelet_length,
            shapelet_dim,
        )
        transient.feature = v
        return 1

cdef struct MetricSubsequenceView:
    Py_ssize_t metric
    SubsequenceView view

cdef struct MetricSubsequence:
    Py_ssize_t metric
    Subsequence subsequence


cdef class MultiMetricShapeletFeatureEngineer(FeatureEngineer):
    cdef Py_ssize_t min_shapelet_size
    cdef Py_ssize_t max_shapelet_size
    cdef SubsequenceMetricList metrics
    cdef bint weighted
    cdef VoseRand vr
    cdef const double[::1] weights

    def __init__(
        self, 
        Py_ssize_t min_shapelet_size, 
        Py_ssize_t max_shapelet_size,
        list metrics,
        const double[::1] weights=None,    
    ):
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.metrics = SubsequenceMetricList(metrics)
        self.weights = weights
        if weights is not None:
            vose_rand_init(&self.vr, weights.shape[0])
            vose_rand_precompute(&self.vr, &weights[0])
            self.weighted = True
        else:
            self.weighted = False
    
    def __dealloc__(self):
        if self.weighted:
            vose_rand_free(&self.vr)

    cdef int reset(self, TSArray X) nogil:
        cdef Py_ssize_t metric
        for metric in range(self.metrics.size):
            self.metrics.reset(metric, X)

        return 1

    cdef Py_ssize_t free_transient_feature(self, Feature *feature) nogil:
        cdef MetricSubsequenceView *msv
        if feature.feature != NULL:
            msv = <MetricSubsequenceView*> feature.feature
            self.metrics.free_transient(msv.metric, &msv.view)
            free(feature.feature)

    cdef Py_ssize_t free_persistent_feature(self, Feature *feature) nogil:
        cdef MetricSubsequence *ms
        if feature.feature != NULL:
            ms = <MetricSubsequence*> feature.feature
            self.metrics.free_persistent(ms.metric, &ms.subsequence)
            free(feature.feature)

    cdef Py_ssize_t init_persistent_feature(
        self, 
        TSArray X,
        Feature *transient, 
        Feature *persistent
    ) nogil:
        cdef MetricSubsequenceView *msv = <MetricSubsequenceView*> transient.feature 
        cdef MetricSubsequence *ms = <MetricSubsequence*> malloc(sizeof(MetricSubsequence))
        
        self.metrics.init_persistent(msv.metric, X, &msv.view, &ms.subsequence)
        ms.metric = msv.metric

        persistent.dim = transient.dim
        persistent.feature = ms
        return 1

    cdef double transient_feature_value(
        self, Feature *feature, TSArray X, Py_ssize_t sample
    ) nogil:
        cdef MetricSubsequenceView *msv = <MetricSubsequenceView*> feature.feature 
        return self.metrics.transient_distance(
            msv.metric, &msv.view, X, sample, NULL
        )

    cdef double persistent_feature_value(
        self, Feature *feature, TSArray X, Py_ssize_t sample
    ) nogil:
        cdef MetricSubsequence *ms = <MetricSubsequence*> feature.feature
        return self.metrics.persistent_distance(
            ms.metric, &ms.subsequence, X, sample, NULL
        )

    cdef Py_ssize_t transient_feature_fill(
        self, 
        Feature *feature, 
        TSArray X, 
        Py_ssize_t sample,
        double[:, :] out,
        Py_ssize_t out_sample,
        Py_ssize_t feature_id,
    ) nogil:
        out[out_sample, feature_id] = self.transient_feature_value(feature, X, sample)
        return 0

    cdef Py_ssize_t persistent_feature_fill(
        self, 
        Feature *feature, 
        TSArray X, 
        Py_ssize_t sample,
        double[:, :] out,
        Py_ssize_t out_sample,
        Py_ssize_t feature_id,
    ) nogil:
        out[out_sample, feature_id] = self.persistent_feature_value(feature, X, sample)
        return 0

    cdef object persistent_feature_to_object(self, Feature *feature):
        cdef MetricSubsequence *ms = <MetricSubsequence*> feature.feature
        return feature.dim, (ms.metric, self.metrics.to_array(ms.metric, &ms.subsequence))

    cdef Py_ssize_t persistent_feature_from_object(self, object object, Feature *feature):
        cdef MetricSubsequence *ms = <MetricSubsequence*> malloc(sizeof(MetricSubsequence))
        dim, (metric, obj) = object
        self.metrics.from_array(metric, &ms.subsequence, obj)
        feature.dim = dim
        feature.feature = ms
        return 0


cdef class RandomMultiMetricShapeletFeatureEngineer(MultiMetricShapeletFeatureEngineer):

    cdef Py_ssize_t n_shapelets

    def __init__(
        self, 
        Py_ssize_t n_shapelets,
        Py_ssize_t min_shapelet_size, 
        Py_ssize_t max_shapelet_size,
        list metrics,
        const double[::1] weights=None,    
    ):
        super().__init__(min_shapelet_size, max_shapelet_size, metrics, weights)
        self.n_shapelets = n_shapelets

    def __reduce__(self):
        return self.__class__, (
            self.n_shapelets, 
            self.min_shapelet_size, 
            self.max_shapelet_size, 
            self.metrics.py_list, 
            np.asarray(self.weights)
        )

    cdef Py_ssize_t get_n_features(self, TSArray X) nogil:
        return self.n_shapelets

    cdef Py_ssize_t next_feature(
        self,
        Py_ssize_t feature_id,
        TSArray X, 
        Py_ssize_t *samples, 
        Py_ssize_t n_samples,
        Feature *transient,
        uint32_t *seed
    ) nogil:
        if feature_id >= self.n_shapelets:
            return -1
        
        cdef Py_ssize_t shapelet_length
        cdef Py_ssize_t shapelet_start
        cdef Py_ssize_t shapelet_index
        cdef Py_ssize_t shapelet_dim
        cdef MetricSubsequenceView *msv = <MetricSubsequenceView*> malloc(
            sizeof(MetricSubsequenceView)
        )

        shapelet_length = rand_int(self.min_shapelet_size, self.max_shapelet_size, seed)
        shapelet_start = rand_int(0, X.shape[2] - shapelet_length, seed)
        shapelet_index = samples[rand_int(0, n_samples, seed)]
        if X.shape[1] > 1:
            shapelet_dim = rand_int(0, X.shape[1], seed)
        else:
            shapelet_dim = 0

        if self.weighted:
            msv.metric = vose_rand_int(&self.vr, seed)
        else:
            msv.metric = rand_int(0, self.metrics.size, seed)

        transient.dim = shapelet_dim
        self.metrics.init_transient(
            msv.metric,
            X,
            &msv.view,
            shapelet_index,
            shapelet_start,
            shapelet_length,
            shapelet_dim,
        )
        transient.feature = msv
        return 1