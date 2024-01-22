# cython: language_level=3
# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: initializedcheck=False

# Authors: Isak Samsten
# License: BSD 3 clause
import numpy as np
cimport numpy as np

from libc.stdlib cimport free, malloc
from libc.math cimport INFINITY, floor, log2, pow
from libc.string cimport memset
from numpy cimport uint32_t

from ..distance._cdistance cimport (
    Subsequence,
    SubsequenceMetric,
    SubsequenceMetricList,
    SubsequenceView,
    Metric,
    dilated_distance_profile,
    scaled_dilated_distance_profile,
)
from ..utils cimport TSArray
from ..utils._misc cimport argsort, to_ndarray_int
from ..utils._stats cimport fast_mean_std
from ..utils._rand cimport (
    RAND_R_MAX,
    VoseRand,
    rand_int,
    rand_uniform,
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


cdef struct DilatedShapelet:
    Py_ssize_t dilation
    Py_ssize_t padding
    Py_ssize_t length
    double* data
    bint is_norm
    double threshold


cdef class DilatedShapeletAttributeGenerator(AttributeGenerator):
    cdef Py_ssize_t n_shapelets
    cdef Py_ssize_t *shapelet_length
    cdef Py_ssize_t n_shapelet_length
    cdef double norm_prob
    cdef double lower_bound
    cdef double upper_bound

    cdef Metric metric
    cdef double* x_buffer
    cdef double* k_buffer
    cdef double* dist_buffer
    cdef Py_ssize_t *arg_buffer

    cdef const Py_ssize_t[:] labels
    cdef const Py_ssize_t[:] samples
    cdef const Py_ssize_t[:] samples_per_label

    def __cinit__(
        self,
        Metric metric,
        Py_ssize_t n_shapelets,
        np.ndarray shapelet_length,
        double norm_prob,
        double lower_bound,
        double upper_bound,
        const Py_ssize_t[:] labels,   # must be the output of np.unique(return_indices=True)
        const Py_ssize_t[:] samples,  # must be np.argsort(labels)
        const Py_ssize_t[:] samples_per_label,  # must be the output of np.unique(return_count=True)
    ):
        self.metric = metric
        self.n_shapelets = n_shapelets
        self.norm_prob = norm_prob
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.n_shapelet_length = shapelet_length.shape[0]
        self.shapelet_length = <Py_ssize_t*> malloc(
            sizeof(Py_ssize_t) * self.n_shapelet_length
        )
        self.x_buffer = NULL
        self.k_buffer = NULL
        self.arg_buffer = NULL
        self.dist_buffer = NULL

        self.labels = labels
        self.samples = samples
        self.samples_per_label = samples_per_label

        cdef Py_ssize_t i
        for i in range(self.n_shapelet_length):
            self.shapelet_length[i] = shapelet_length[i]

    def __reduce__(self):
        return self.__class__, (
            self.metric,
            self.n_shapelets,
            to_ndarray_int(self.shapelet_length, self.n_shapelet_length),
            self.norm_prob,
            self.lower_bound,
            self.upper_bound,
            self.labels.base if self.labels is not None else None,
            self.samples.base if self.samples is not None else None,
            self.samples_per_label.base if self.samples_per_label is not None else None,
        )

    def __dealloc__(self):
        self._free()
        if self.shapelet_length != NULL:
            free(self.shapelet_length)
            self.shapelet_length = NULL


    cdef void _free(self) noexcept nogil:
        if self.x_buffer != NULL:
            free(self.x_buffer)
            self.x_buffer = NULL

        if self.k_buffer != NULL:
            free(self.k_buffer)
            self.k_buffer = NULL

        if self.dist_buffer != NULL:
            free(self.dist_buffer)
            self.dist_buffer = NULL

        if self.arg_buffer != NULL:
            free(self.arg_buffer)
            self.arg_buffer = NULL

    cdef int reset(self, TSArray X) noexcept nogil:
        self._free()
        self.x_buffer = <double*> malloc(sizeof(double) * X.shape[2] + 1)
        self.k_buffer = <double*> malloc(sizeof(double) * X.shape[2] + 1)
        self.dist_buffer = <double*> malloc(sizeof(double) * X.shape[2] + 1)
        self.arg_buffer = <Py_ssize_t*> malloc(sizeof(double) * X.shape[2] + 1)
        self.metric.reset(X, X)

        return 1

    cdef Py_ssize_t get_n_attributess(self, TSArray X) noexcept nogil:
        return self.n_shapelets

    cdef Py_ssize_t get_n_outputs(self, TSArray X) noexcept nogil:
        return self.get_n_attributess(X) * 3  # min, argmin, SO(S, X)

    cdef Py_ssize_t next_attribute(
        self,
        Py_ssize_t attribute_id,
        TSArray X,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        Attribute *transient,
        uint32_t *seed
    ) noexcept nogil:
        cdef double mean, std
        cdef Py_ssize_t i, start, index, dim
        cdef DilatedShapelet *shapelet = <DilatedShapelet*> malloc(
            sizeof(DilatedShapelet)
        )

        shapelet.length = self.shapelet_length[
            rand_int(0, self.n_shapelet_length, seed)
        ]

        # TODO: ensure that the shapelet length is odd
        # so that the dilation/padding works as expected. Drop +1 above.
        # if shapelet.length > X.shape[2]:
        #     shapelet.length = X.shape[2]

        shapelet.data = <double*> malloc(sizeof(double) * shapelet.length)

        if X.shape[2] > 1:
            shapelet.dilation = <Py_ssize_t> floor(
                pow(
                    2,
                    rand_uniform(
                        0, max(0, log2((X.shape[2] - 1) / (shapelet.length - 1))), seed
                    ),
                )
            )
        else:
            shapelet.dilation = 1

        cdef Py_ssize_t dilated_length = (shapelet.length - 1) * shapelet.dilation + 1
        start = rand_int(0, max(0, X.shape[2] - dilated_length), seed)
        index = samples[rand_int(0, n_samples, seed)]
        if X.shape[1] > 1:
            dim = rand_int(0, X.shape[1], seed)
        else:
            dim = 0

        shapelet.padding = dilated_length // 2
        shapelet.is_norm = rand_uniform(0, 1, seed) < self.norm_prob

        # cdef const double* sample = &X[index, dim, start]
        cdef Py_ssize_t j = 0
        for i from start <= i < start + dilated_length by shapelet.dilation:
            if i < X.shape[2]:
                shapelet.data[j] = X[index, dim, i]
            else:
                shapelet.data[j] = 0
            j += 1

        if shapelet.is_norm:
            fast_mean_std(shapelet.data, shapelet.length, &mean, &std)
            if std == 0:
                std = 1

            for i in range(shapelet.length):
                shapelet.data[i] = (shapelet.data[i] - mean) / std

        if self.samples is not None:
            index = self._sample_other_same_label(index, seed)

        cdef Py_ssize_t n_distances = self._get_distance_profile(
            shapelet, &X[index, dim, 0], X.shape[2], False
        )

        for i in range(n_distances):
            self.arg_buffer[i] = i
        argsort(self.dist_buffer, self.arg_buffer, n_distances)

        cdef Py_ssize_t lower = <Py_ssize_t> floor(n_distances * self.lower_bound)
        cdef Py_ssize_t upper = <Py_ssize_t> floor(n_distances * self.upper_bound)
        cdef Py_ssize_t rand_threshold = self.arg_buffer[rand_int(lower, upper, seed)]
        shapelet.threshold = self.dist_buffer[rand_threshold]
        transient.dim = dim
        transient.attribute = shapelet
        return 0


    cdef Py_ssize_t free_transient(self, Attribute *attribute) noexcept nogil:
        return self.free_persistent(attribute)

    cdef Py_ssize_t free_persistent(self, Attribute *attribute) noexcept nogil:
        cdef DilatedShapelet *shapelet
        if attribute.attribute != NULL:
            shapelet = <DilatedShapelet*> attribute.attribute
            if shapelet.data != NULL:
                free(shapelet.data)
            free(attribute.attribute)
            attribute.attribute = NULL
        return 0

    # NOTE: We move ownership of `transient.attribute` to `persistent.attribute`.
    cdef Py_ssize_t init_persistent(
        self,
        TSArray X,
        Attribute *transient,
        Attribute *persistent
    ) noexcept nogil:
        persistent.dim = transient.dim
        persistent.attribute = transient.attribute
        transient.attribute = NULL  # No need to free it
        return 0

    cdef object persistent_to_object(self, Attribute *attribute):
        cdef Py_ssize_t j
        cdef DilatedShapelet *shapelet = <DilatedShapelet*> attribute.attribute

        data = np.empty(shapelet.length, dtype=float)
        for j in range(shapelet.length):
            data[j] = shapelet.data[j]

        return attribute.dim, (
            shapelet.dilation,
            shapelet.padding,
            shapelet.length,
            data,
            shapelet.is_norm,
            shapelet.threshold,
        )

    cdef Py_ssize_t persistent_from_object(self, object object, Attribute *attribute):
        dim, (dilation, padding, length, data, is_norm, threshold) = object

        cdef DilatedShapelet *shapelet = <DilatedShapelet*> malloc(sizeof(DilatedShapelet))
        shapelet.dilation = dilation
        shapelet.padding = padding
        shapelet.length = length
        shapelet.data = <double*> malloc(sizeof(double) * length)
        shapelet.threshold = threshold
        shapelet.is_norm = is_norm

        cdef Py_ssize_t i
        for i in range(length):
            shapelet.data[i] = data[i]

        attribute.attribute = shapelet
        attribute.dim = dim
        return 0

    cdef Py_ssize_t transient_fill(
        self,
        Attribute *attribute,
        TSArray X,
        Py_ssize_t sample,
        double[:, :] out,
        Py_ssize_t out_sample,
        Py_ssize_t attribute_id,
    ) noexcept nogil:
        cdef double minval = INFINITY
        cdef double minarg = -1
        cdef double so = 0
        cdef Py_ssize_t i
        cdef DilatedShapelet* shapelet = <DilatedShapelet*> attribute.attribute

        cdef Py_ssize_t n_distances = self._get_distance_profile(
            shapelet, &X[sample, attribute.dim, 0], X.shape[2], False
        )

        # If we got no distances, we need to compute the full
        # distance profile to find the min.
        if n_distances == 0:
            n_distances = self._get_distance_profile(
                shapelet, &X[sample, attribute.dim, 0], X.shape[2], False
            )

        for i in range(n_distances):
            if self.dist_buffer[i] < minval:
                minval = self.dist_buffer[i]
                minarg = i
            if self.dist_buffer[i] < shapelet.threshold:
                so += 1

        cdef Py_ssize_t attribute_offset = attribute_id * 3
        out[out_sample, attribute_offset] = minval
        out[out_sample, attribute_offset + 1] = minarg
        out[out_sample, attribute_offset + 2] = so
        return 0

    cdef Py_ssize_t persistent_fill(
        self,
        Attribute *attribute,
        TSArray X,
        Py_ssize_t sample,
        double[:, :] out,
        Py_ssize_t out_sample,
        Py_ssize_t out_attribute,
    ) noexcept nogil:
        return self.transient_fill(
            attribute, X, sample, out, out_sample, out_attribute
        )

    cdef double transient_value(
        self,
        Attribute *attribute,
        TSArray X,
        Py_ssize_t sample
    ) noexcept nogil:
        cdef double minval = INFINITY
        cdef Py_ssize_t so = 0
        cdef DilatedShapelet* shapelet = <DilatedShapelet*> attribute.attribute
        cdef Py_ssize_t n_distances = self._get_distance_profile(
            shapelet, &X[sample, attribute.dim, 0], X.shape[2], True
        )
        if n_distances == 0:
            n_distances = self._get_distance_profile(
                shapelet, &X[sample, attribute.dim, 0], X.shape[2], False
            )

        for i in range(n_distances):
            if self.dist_buffer[i] < minval:
                minval = self.dist_buffer[i]
            if self.dist_buffer[i] < shapelet.threshold:
                so += 1

        return minval / so

    cdef double persistent_value(
        self,
        Attribute *attribute,
        TSArray X,
        Py_ssize_t sample
    ) noexcept nogil:
        return self.transient_value(attribute, X, sample)

    # Guarded by if self.samples is not None
    cdef Py_ssize_t _sample_other_same_label(
        self, Py_ssize_t sample, uint32_t *seed
    ) noexcept nogil:
        cdef Py_ssize_t label = self.labels[sample]
        cdef Py_ssize_t label_start = 0
        cdef Py_ssize_t label_end = 0
        cdef Py_ssize_t i
        for i in range(label):
            label_start += self.samples_per_label[i]
            label_end += self.samples_per_label[i]

        label_end += self.samples_per_label[label]
        return self.samples[rand_int(label_start, label_end, seed)]

    cdef Py_ssize_t _get_distance_profile(
        self,
        DilatedShapelet* shapelet,
        const double* x,
        Py_ssize_t x_len,
        bint early_abandon,
    ) noexcept nogil:
        if shapelet.is_norm:
            return scaled_dilated_distance_profile(
                1,
                shapelet.dilation,
                shapelet.padding,
                shapelet.data,
                shapelet.length,
                x,
                x_len,
                self.metric,
                self.x_buffer,
                self.k_buffer,
                shapelet.threshold if early_abandon else INFINITY,
                self.dist_buffer,
            )
        else:
            return dilated_distance_profile(
                1,
                shapelet.dilation,
                shapelet.padding,
                shapelet.data,
                shapelet.length,
                x,
                x_len,
                self.metric,
                self.x_buffer,
                self.k_buffer,
                shapelet.threshold if early_abandon else INFINITY,
                self.dist_buffer,
            )


cdef inline Py_ssize_t _max_exponent(
    Py_ssize_t n_timestep, Py_ssize_t kernel_size
) noexcept nogil:
    cdef Py_ssize_t max_exponent = <Py_ssize_t> floor(
        log2((n_timestep - 1) / <double> (kernel_size - 1))
    )
    if max_exponent < 0:
        max_exponent = 0
    max_exponent += 1

    return max_exponent


cdef struct Cdist:
    bint is_norm
    Py_ssize_t max_exponent

    # size: max_exponent * n_shapelets
    double *thresholds

    # size: shapelet_size * n_shapelets
    double *data


cdef class CastorSummarizer:
    cdef Py_ssize_t n_timestep
    cdef Py_ssize_t n_shapelets
    cdef bint soft_min, soft_max, soft_threshold
    cdef double *min_values
    cdef double *max_values
    cdef double *min_so_values

    def __cinit__(self, *args, **kwargs):
        self.min_values = NULL
        self.max_values = NULL
        self.min_so_values = NULL

    def __init__(self, soft_min=True, soft_max=False, soft_threshold=True):
        self.soft_min = soft_min
        self.soft_max = soft_max
        self.soft_threshold = soft_threshold

    def __reduce__(self):
        return self.__class__, (self.soft_min, self.soft_max, self.soft_threshold)

    def __dealloc__(self):
        self._free()

    cdef Py_ssize_t get_n_features(self) noexcept nogil:
        return 3

    cdef void _free(self) noexcept nogil:
        if self.min_values != NULL:
            free(self.min_values)
            self.min_values = NULL
        if self.max_values != NULL:
            free(self.max_values)
            self.max_values = NULL
        if self.min_so_values != NULL:
            free(self.min_so_values)
            self.min_so_values = NULL

    cdef int reset(self, Py_ssize_t n_timestep, Py_ssize_t n_shapelets) noexcept nogil:
        self._free()
        self.n_timestep = n_timestep
        self.n_shapelets = n_shapelets
        self.min_values = <double*> malloc(sizeof(double) * self.n_shapelets)
        self.max_values = <double*> malloc(sizeof(double) * self.n_shapelets)
        self.min_so_values = <double*> malloc(sizeof(double) * self.n_shapelets)

    cdef void fill(
        self,
        Py_ssize_t attribute_offset,
        Py_ssize_t exponent,
        Cdist *cdist,
        double *dist_buffer,
        double[:] out,
    ) noexcept nogil:
        cdef Py_ssize_t i, min_index, max_index
        cdef double min_value, max_value

        memset(self.min_values, 0, sizeof(double) * self.n_shapelets)
        memset(self.max_values, 0, sizeof(double) * self.n_shapelets)
        memset(self.min_so_values, 0, sizeof(double) * self.n_shapelets)

        for i in range(self.n_timestep):
            self._compute_attributes(
                i,
                self.n_timestep,
                exponent,
                cdist.max_exponent,
                dist_buffer,
                cdist.thresholds,
                &min_index,
                &min_value,
                &max_index,
                &max_value,
                self.min_so_values,
            )

            self.max_values[max_index] += max_value if self.soft_max else 1
            self.min_values[min_index] += min_value if self.soft_min else 1
            if (
                not self.soft_threshold and
                self.min_values[min_index] < cdist.thresholds[min_index * cdist.max_exponent + exponent]
            ):
                self.min_so_values[min_index] += 1

        cdef Py_ssize_t shapelet_attribute_offset = attribute_offset + exponent * self.n_shapelets * 3
        for i in range(self.n_shapelets):
            out[shapelet_attribute_offset + i * 3 + 0] = self.min_values[i]
            out[shapelet_attribute_offset + i * 3 + 1] = self.max_values[i]
            out[shapelet_attribute_offset + i * 3 + 2] = self.min_so_values[i]

    cdef void _compute_attributes(
        self,
        Py_ssize_t offset,
        Py_ssize_t stride,
        Py_ssize_t exponent,
        Py_ssize_t max_exponent,
        double *values,
        double *thresholds,
        Py_ssize_t *min_index,
        double *min_value,
        Py_ssize_t *max_index,
        double *max_value,
        double *min_so_values,
    ) noexcept nogil:
        cdef Py_ssize_t i
        cdef double value, threshold
        min_value[0] = INFINITY
        max_value[0] = -INFINITY
        for i in range(self.n_shapelets):
            value = values[offset + i * stride]
            threshold = thresholds[i * max_exponent + exponent]

            if value < min_value[0]:
                min_value[0] = value
                min_index[0] = i
            if value < threshold and self.soft_threshold:
                min_so_values[i] += 1
            if value > max_value[0]:
                max_value[0] = value
                max_index[0] = i

cdef class CastorAttributeGenerator(AttributeGenerator):
    cdef Py_ssize_t n_shapelets
    cdef Py_ssize_t shapelet_size
    cdef Py_ssize_t n_groups
    cdef double normalize_prob
    cdef double lower
    cdef double upper

    cdef CastorSummarizer summarizer
    cdef Metric metric

    # Temporary buffer to store dilated distance
    cdef double *dist_buffer
    cdef Py_ssize_t *arg_buffer

    cdef double *x_buffer
    cdef double *s_buffer

    # Temporary buffer to store kernel values
    cdef double *max_values
    cdef double *min_values
    cdef double *min_so_values
    cdef double *max_so_values

    # random values and incremental indices to draw random samples from
    cdef double *rnd_value

    cdef const Py_ssize_t[:] labels
    cdef const Py_ssize_t[:] samples
    cdef const Py_ssize_t[:] samples_per_label

    def __cinit__(
        self,
        Py_ssize_t n_groups,
        Py_ssize_t n_shapelets,
        Py_ssize_t shapelet_size,
        double normalize_prob,
        double lower,
        double upper,
        Metric metric,
        CastorSummarizer summarizer,
        const Py_ssize_t[:] labels,   # must be the output of np.unique(return_indices=True)
        const Py_ssize_t[:] samples,  # must be np.argsort(labels)
        const Py_ssize_t[:] samples_per_label,  # must be the output of np.unique(return_count=True)
    ):
        self.n_groups = n_groups
        self.n_shapelets = n_shapelets
        self.shapelet_size = shapelet_size
        self.normalize_prob = normalize_prob
        self.lower = lower
        self.upper = upper
        self.metric = metric
        self.summarizer = summarizer

        self.rnd_value = NULL
        self.dist_buffer = NULL
        self.arg_buffer = NULL
        self.x_buffer = NULL
        self.s_buffer = NULL
        self.max_values = NULL
        self.min_values = NULL
        self.min_so_values = NULL
        self.max_so_values = NULL
        self.labels = labels
        self.samples = samples
        self.samples_per_label = samples_per_label

    def __reduce__(self):
        return self.__class__, (
            self.n_groups,
            self.n_shapelets,
            self.shapelet_size,
            self.normalize_prob,
            self.lower,
            self.upper,
            self.metric,
            self.summarizer,
            self.labels.base if self.labels is not None else None,
            self.samples.base if self.samples is not None else None,
            self.samples_per_label.base if self.samples_per_label is not None else None,
        )

    def __dealloc__(self):
        self._free()

    cdef void _free(self) noexcept nogil:
        if self.arg_buffer != NULL:
            free(self.arg_buffer)
            self.arg_buffer = NULL
        if self.dist_buffer != NULL:
            free(self.dist_buffer)
            self.dist_buffer = NULL
        if self.x_buffer != NULL:
            free(self.x_buffer)
            self.x_buffer = NULL
        if self.s_buffer != NULL:
            free(self.s_buffer)
            self.s_buffer = NULL
        if self.min_values != NULL:
            free(self.min_values)
            self.min_values = NULL
        if self.max_values != NULL:
            free(self.max_values)
            self.max_values = NULL
        if self.min_so_values != NULL:
            free(self.min_so_values)
            self.min_so_values = NULL
        if self.max_so_values != NULL:
            free(self.max_so_values)
            self.max_so_values = NULL
        if self.rnd_value != NULL:
            free(self.rnd_value)
            self.rnd_value = NULL

    cdef int reset(self, TSArray X) noexcept nogil:
        self._free()
        self.dist_buffer = <double*> malloc(
            sizeof(double) * X.shape[2] * self.n_shapelets
        )
        self.arg_buffer = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * X.shape[2])
        self.x_buffer = <double*> malloc(sizeof(double) * self.shapelet_size)
        self.s_buffer = <double*> malloc(sizeof(double) * self.shapelet_size)
        self.min_values = <double*> malloc(sizeof(double) * self.n_shapelets)
        self.max_values = <double*> malloc(sizeof(double) * self.n_shapelets)
        self.min_so_values = <double*> malloc(sizeof(double) * self.n_shapelets)
        self.max_so_values = <double*> malloc(sizeof(double) * self.n_shapelets)
        self.rnd_value = <double*> malloc(sizeof(double) * X.shape[0])

        self.metric.reset(X, X)
        self.summarizer.reset(X.shape[2], self.n_shapelets)
        return 0

    cdef Py_ssize_t get_n_attributess(self, TSArray X) noexcept nogil:
        return self.n_groups

    # n_shapelets * max_exponent * n_shapelets * 3 features per time series.
    cdef Py_ssize_t get_n_outputs(self, TSArray X) noexcept nogil:
        cdef Py_ssize_t max_exponent = _max_exponent(X.shape[2], self.shapelet_size)
        return (
            self.get_n_attributess(X) * max_exponent * self.n_shapelets * self.summarizer.get_n_features()
        )

    cdef Py_ssize_t _get_distance_profile(
        self,
        Py_ssize_t dilation,
        Py_ssize_t padding,
        const double *s,
        Py_ssize_t s_len,
        const double *x,
        Py_ssize_t x_len,
        bint is_norm,
        double *dist_buffer,
    ) noexcept nogil:
        if is_norm:
            return scaled_dilated_distance_profile(
                1,
                dilation,
                padding,
                s,
                s_len,
                x,
                x_len,
                self.metric,
                self.x_buffer,
                self.s_buffer,
                INFINITY,
                dist_buffer,
            )
        else:
            return dilated_distance_profile(
                1,
                dilation,
                padding,
                s,
                s_len,
                x,
                x_len,
                self.metric,
                self.x_buffer,
                self.s_buffer,
                INFINITY,
                dist_buffer,
            )

    cdef Py_ssize_t next_attribute(
        self,
        Py_ssize_t attribute_id,
        TSArray X,
        Py_ssize_t *samples,
        Py_ssize_t n_samples,
        Attribute *transient,
        uint32_t *seed
    ) noexcept nogil:
        cdef Cdist *cdist = <Cdist*> malloc(sizeof(Cdist))
        cdef Py_ssize_t i, j, k, m, n
        cdef Py_ssize_t dim, start, dilation, dilated_size, padding, n_distances
        cdef Py_ssize_t lower, upper
        cdef double mean, std
        cdef Py_ssize_t max_exponent = _max_exponent(X.shape[2], self.shapelet_size)

        for i in range(n_samples):
            self.rnd_value[i] = rand_uniform(0, 1, seed)

        argsort(self.rnd_value, samples, n_samples)

        # shapelet[0, 0], ..., shapelet[n_shapelets, max_exponent]
        cdef double *data = <double*> malloc(
            sizeof(double) * self.n_shapelets * max_exponent * self.shapelet_size
        )
        cdef double *thresholds = <double*> malloc(
            sizeof(double) * max_exponent * self.n_shapelets
        )
        cdef double *shapelet
        cdef double *x

        if X.shape[1] > 1:
            dim = rand_int(0, X.shape[1], seed)
        else:
            dim = 0

        cdist.is_norm = rand_uniform(0, 1, seed) < self.normalize_prob
        cdist.max_exponent = max_exponent

        # Draw samples per shapelet/dilation pair not per shapelet.
        for i in range(self.n_shapelets):
            for k in range(max_exponent):
                j = samples[(i + k) % n_samples]
                shapelet = (
                    data
                    + i * self.shapelet_size * max_exponent
                    + k * self.shapelet_size
                )

                dilation = <Py_ssize_t> pow(2, k)
                dilated_size = (self.shapelet_size - 1) * dilation + 1
                start = rand_int(0, max(0, X.shape[2] - dilated_size), seed)
                padding = dilated_size // 2

                n = 0
                for m from start <= m < start + dilated_size by dilation:
                    if m < X.shape[2]:
                        shapelet[n] = X[j, dim, m]
                    else:
                        shapelet[n] = 0.0
                    n += 1

                if cdist.is_norm:
                    fast_mean_std(shapelet, self.shapelet_size, &mean, &std)
                    if std == 0.0:
                        std = 1.0

                    for m in range(self.shapelet_size):
                        shapelet[m] = (shapelet[m] - mean) / std

                if self.samples is not None:
                    x = &X[self._sample_other_same_label(j, seed), dim, 0]
                else:
                    x = &X[j, dim, 0]

                n_distances = self._get_distance_profile(
                    dilation,
                    padding,
                    shapelet,
                    self.shapelet_size,
                    x,
                    X.shape[2],
                    cdist.is_norm,
                    self.dist_buffer,
                )

                for m in range(n_distances):
                    self.arg_buffer[m] = m
                argsort(self.dist_buffer, self.arg_buffer, n_distances)

                lower = <Py_ssize_t> floor(n_distances * self.lower)
                upper = <Py_ssize_t> floor(n_distances * self.upper)

                thresholds[i * max_exponent + k] = self.dist_buffer[
                    rand_int(lower, upper, seed)
                ]

        cdist.thresholds = thresholds
        cdist.data = data

        transient.dim = dim
        transient.attribute = cdist
        return 0

    # NOTE: We move ownership of `transient.attribute` to `persistent.attribute`.
    cdef Py_ssize_t init_persistent(
        self,
        TSArray X,
        Attribute *transient,
        Attribute *persistent
    ) noexcept nogil:
        persistent.dim = transient.dim
        persistent.attribute = transient.attribute
        transient.attribute = NULL  # No need to free it
        return 0

    cdef Py_ssize_t free_persistent(self, Attribute *attribute) noexcept nogil:
        cdef Cdist *cdist
        if attribute.attribute != NULL:
            cdist = <Cdist*> attribute.attribute
            if cdist.data != NULL:
                free(cdist.data)
            if cdist.thresholds != NULL:
                free(cdist.thresholds)

            free(attribute.attribute)
            attribute.attribute = NULL
        return 0

    cdef Py_ssize_t free_transient(self, Attribute *attribute) noexcept nogil:
        return self.free_persistent(attribute)

    cdef object persistent_to_object(self, Attribute *attribute):
        cdef Py_ssize_t i
        cdef Cdist *cdist = <Cdist*> attribute.attribute

        data = np.empty(
            self.shapelet_size * cdist.max_exponent * self.n_shapelets, dtype=float
        )
        thresholds = np.empty(cdist.max_exponent * self.n_shapelets, dtype=float)

        for i in range(data.shape[0]):
            data[i] = cdist.data[i]

        for i in range(thresholds.shape[0]):
            thresholds[i] = cdist.thresholds[i]

        return attribute.dim, (
            cdist.is_norm,
            cdist.max_exponent,
            data,
            thresholds,
        )

    cdef Py_ssize_t persistent_from_object(self, object obj, Attribute *attribute):
        dim, (is_norm, max_exponent, data, thresholds) = obj
        cdef Cdist *cdist = <Cdist*> malloc(sizeof(Cdist))

        cdist.is_norm = is_norm
        cdist.max_exponent = max_exponent
        cdist.data = <double*> malloc(sizeof(double) * data.shape[0])
        cdist.thresholds = <double*> malloc(sizeof(double) * thresholds.shape[0])
        cdef Py_ssize_t i

        for i in range(data.shape[0]):
            cdist.data[i] = data[i]

        for i in range(thresholds.shape[0]):
            cdist.thresholds[i] = thresholds[i]

        attribute.attribute = cdist
        attribute.dim = dim
        return 0

    cdef Py_ssize_t transient_fill(
        self,
        Attribute *attribute,
        TSArray X,
        Py_ssize_t sample,
        double[:, :] out,
        Py_ssize_t out_sample,
        Py_ssize_t attribute_id,
    ) noexcept nogil:
        cdef Py_ssize_t i, exponent, padding, dilation
        cdef double min_value, max_value
        cdef Py_ssize_t min_index, max_index

        cdef Cdist *cdist = <Cdist*> attribute.attribute
        cdef double *shapelet

        # Place the pointer inside correct attribute group, as given by attribute_id
        cdef Py_ssize_t shapelet_attribute_offset
        cdef Py_ssize_t attribute_offset = (
            attribute_id * self.n_shapelets * cdist.max_exponent * 3
        )

        for exponent in range(cdist.max_exponent):
            dilation = <Py_ssize_t> pow(2, exponent)
            padding = <Py_ssize_t> ((self.shapelet_size - 1) * dilation) // 2

            for i in range(self.n_shapelets):
                shapelet = (
                    cdist.data
                    + i * self.shapelet_size * cdist.max_exponent
                    + exponent * self.shapelet_size
                )
                self._get_distance_profile(
                    dilation,
                    padding,
                    shapelet,
                    self.shapelet_size,
                    &X[sample, attribute.dim, 0],
                    X.shape[2],
                    cdist.is_norm,
                    self.dist_buffer + i * X.shape[2],
                )

            self.summarizer.fill(attribute_offset, exponent, cdist, self.dist_buffer, out[out_sample])

    cdef Py_ssize_t persistent_fill(
        self,
        Attribute *attribute,
        TSArray X,
        Py_ssize_t sample,
        double[:, :] out,
        Py_ssize_t out_sample,
        Py_ssize_t out_attribute,
    ) noexcept nogil:
        return self.transient_fill(
            attribute, X, sample, out, out_sample, out_attribute
        )

    # Guarded by if self.samples is not None
    cdef Py_ssize_t _sample_other_same_label(
        self, Py_ssize_t sample, uint32_t *seed
    ) noexcept nogil:
        cdef Py_ssize_t label = self.labels[sample]
        cdef Py_ssize_t label_start = 0
        cdef Py_ssize_t label_end = 0
        cdef Py_ssize_t i
        for i in range(label):
            label_start += self.samples_per_label[i]
            label_end += self.samples_per_label[i]

        label_end += self.samples_per_label[label]
        return self.samples[rand_int(label_start, label_end, seed)]

    cdef void _compute_attributes(
        self,
        Py_ssize_t offset,
        Py_ssize_t stride,
        Py_ssize_t exponent,
        Py_ssize_t max_exponent,
        double *values,
        double *thresholds,
        Py_ssize_t *min_index,
        double *min_value,
        Py_ssize_t *max_index,
        double *max_value,
        double *min_so_values,
    ) noexcept nogil:
        cdef Py_ssize_t i
        cdef double value, threshold
        min_value[0] = INFINITY
        max_value[0] = -INFINITY
        for i in range(self.n_shapelets):
            value = values[offset + i * stride]
            threshold = thresholds[i * max_exponent + exponent]

            if value < min_value[0]:
                min_value[0] = value
                min_index[0] = i
            if value < threshold:
                min_so_values[i] += 1
            if value > max_value[0]:
                max_value[0] = value
                max_index[0] = i
