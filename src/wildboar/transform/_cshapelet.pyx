# cython: language_level=3
# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: initializedcheck=False

# Authors: Isak Samsten
# License: BSD 3 clause
import numpy as np

from libc.stdlib cimport free, malloc
from numpy cimport uint32_t

from ..distance._cdistance cimport (
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
from ._attr_gen cimport Attribute, AttributeGenerator


cdef class ShapeletAttributeGenerator(AttributeGenerator):

    cdef Py_ssize_t min_shapelet_size
    cdef Py_ssize_t max_shapelet_size

    cdef readonly SubsequenceMetric metric

    def __init__(self, metric, min_shapelet_size, max_shapelet_size):
        self.metric = metric
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size

    cdef int reset(self, TSArray X) noexcept nogil:
        self.metric.reset(X)
        return 1

    cdef Py_ssize_t free_transient(self, Attribute *attribute) noexcept nogil:
        if attribute.attribute != NULL:
            self.metric.free_transient(<SubsequenceView*> attribute.attribute)
            free(attribute.attribute)

    cdef Py_ssize_t free_persistent(self, Attribute *attribute) noexcept nogil:
        cdef Subsequence *s
        if attribute.attribute != NULL:
            self.metric.free_persistent(<Subsequence*> attribute.attribute)
            free(attribute.attribute)

    cdef Py_ssize_t init_persistent(
        self,
        TSArray X,
        Attribute *transient,
        Attribute *persistent
    ) noexcept nogil:
        cdef SubsequenceView *v = <SubsequenceView*> transient.attribute
        cdef Subsequence *s = <Subsequence*> malloc(sizeof(Subsequence))
        self.metric.init_persistent(X, v, s)
        persistent.dim = transient.dim
        persistent.attribute = s
        return 1

    cdef double transient_value(
        self,
        Attribute *attribute,
        TSArray X,
        Py_ssize_t sample
    ) noexcept nogil:
        return self.metric.transient_distance(
            <SubsequenceView*> attribute.attribute, X, sample
        )

    cdef double persistent_value(
        self,
        Attribute *attribute,
        TSArray X,
        Py_ssize_t sample
    ) noexcept nogil:
        return self.metric.persistent_distance(
            <Subsequence*> attribute.attribute, X, sample
        )

    cdef Py_ssize_t transient_fill(
        self,
        Attribute *attribute,
        TSArray X,
        Py_ssize_t sample,
        double[:, :] out,
        Py_ssize_t out_sample,
        Py_ssize_t attribute_id,
    ) noexcept nogil:
        out[out_sample, attribute_id] = self.metric.transient_distance(
            <SubsequenceView*> attribute.attribute, X, sample
        )
        return 0

    cdef Py_ssize_t persistent_fill(
        self,
        Attribute *attribute,
        TSArray X,
        Py_ssize_t sample,
        double[:, :] out,
        Py_ssize_t out_sample,
        Py_ssize_t attribute_id,
    ) noexcept nogil:
        out[out_sample, attribute_id] = self.metric.persistent_distance(
            <Subsequence*> attribute.attribute, X, sample
        )
        return 0

    cdef object persistent_to_object(self, Attribute *attribute):
        return attribute.dim, self.metric.to_array(<Subsequence*>attribute.attribute)

    cdef Py_ssize_t persistent_from_object(self, object object, Attribute *attribute):
        cdef Subsequence *s = <Subsequence*> malloc(sizeof(Subsequence))
        dim, obj = object
        self.metric.from_array(s, obj)
        attribute.dim = dim
        attribute.attribute = s
        return 0

cdef class RandomShapeletAttributeGenerator(ShapeletAttributeGenerator):

    cdef Py_ssize_t n_shapelets

    def __init__(
        self, metric, min_shapelet_size, max_shapelet_size, n_shapelets
    ):
        super().__init__(metric, min_shapelet_size, max_shapelet_size)
        self.n_shapelets = n_shapelets

    cdef Py_ssize_t get_n_attributess(self, TSArray X) noexcept nogil:
        return self.n_shapelets

    cdef Py_ssize_t next_attribute(
        self,
        Py_ssize_t attribute_id,
        TSArray X,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        Attribute *transient,
        uint32_t *random_seed
    ) noexcept nogil:
        if attribute_id >= self.n_shapelets:
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
        transient.attribute = v
        return 1

cdef struct MetricSubsequenceView:
    Py_ssize_t metric
    SubsequenceView view

cdef struct MetricSubsequence:
    Py_ssize_t metric
    Subsequence subsequence


cdef class MultiMetricShapeletAttributeGenerator(AttributeGenerator):
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

    cdef int reset(self, TSArray X) noexcept nogil:
        cdef Py_ssize_t metric
        for metric in range(self.metrics.size):
            self.metrics.reset(metric, X)

        return 1

    cdef Py_ssize_t free_transient(self, Attribute *attribute) noexcept nogil:
        cdef MetricSubsequenceView *msv
        if attribute.attribute != NULL:
            msv = <MetricSubsequenceView*> attribute.attribute
            self.metrics.free_transient(msv.metric, &msv.view)
            free(attribute.attribute)

    cdef Py_ssize_t free_persistent(self, Attribute *attribute) noexcept nogil:
        cdef MetricSubsequence *ms
        if attribute.attribute != NULL:
            ms = <MetricSubsequence*> attribute.attribute
            self.metrics.free_persistent(ms.metric, &ms.subsequence)
            free(attribute.attribute)

    cdef Py_ssize_t init_persistent(
        self,
        TSArray X,
        Attribute *transient,
        Attribute *persistent
    ) noexcept nogil:
        cdef MetricSubsequenceView *msv = <MetricSubsequenceView*> transient.attribute
        cdef MetricSubsequence *ms = <MetricSubsequence*> malloc(
            sizeof(MetricSubsequence)
        )

        self.metrics.init_persistent(msv.metric, X, &msv.view, &ms.subsequence)
        ms.metric = msv.metric

        persistent.dim = transient.dim
        persistent.attribute = ms
        return 1

    cdef double transient_value(
        self, Attribute *attribute, TSArray X, Py_ssize_t sample
    ) noexcept nogil:
        cdef MetricSubsequenceView *msv = <MetricSubsequenceView*> attribute.attribute
        return self.metrics.transient_distance(
            msv.metric, &msv.view, X, sample, NULL
        )

    cdef double persistent_value(
        self, Attribute *attribute, TSArray X, Py_ssize_t sample
    ) noexcept nogil:
        cdef MetricSubsequence *ms = <MetricSubsequence*> attribute.attribute
        return self.metrics.persistent_distance(
            ms.metric, &ms.subsequence, X, sample, NULL
        )

    cdef Py_ssize_t transient_fill(
        self,
        Attribute *attribute,
        TSArray X,
        Py_ssize_t sample,
        double[:, :] out,
        Py_ssize_t out_sample,
        Py_ssize_t attribute_id,
    ) noexcept nogil:
        out[out_sample, attribute_id] = self.transient_value(attribute, X, sample)
        return 0

    cdef Py_ssize_t persistent_fill(
        self,
        Attribute *attribute,
        TSArray X,
        Py_ssize_t sample,
        double[:, :] out,
        Py_ssize_t out_sample,
        Py_ssize_t attribute_id,
    ) noexcept nogil:
        out[out_sample, attribute_id] = self.persistent_value(attribute, X, sample)
        return 0

    cdef object persistent_to_object(self, Attribute *attribute):
        cdef MetricSubsequence *ms = <MetricSubsequence*> attribute.attribute
        return attribute.dim, (
            ms.metric, self.metrics.to_array(ms.metric, &ms.subsequence)
        )

    cdef Py_ssize_t persistent_from_object(self, object object, Attribute *attribute):
        cdef MetricSubsequence *ms = <MetricSubsequence*> malloc(
            sizeof(MetricSubsequence)
        )
        dim, (metric, obj) = object
        self.metrics.from_array(metric, &ms.subsequence, obj)
        attribute.dim = dim
        attribute.attribute = ms
        return 0


cdef class RandomMultiMetricShapeletAttributeGenerator(
    MultiMetricShapeletAttributeGenerator
):
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

    cdef Py_ssize_t get_n_attributess(self, TSArray X) noexcept nogil:
        return self.n_shapelets

    cdef Py_ssize_t next_attribute(
        self,
        Py_ssize_t attribute_id,
        TSArray X,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        Attribute *transient,
        uint32_t *seed
    ) noexcept nogil:
        if attribute_id >= self.n_shapelets:
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
        transient.attribute = msv
        return 1
