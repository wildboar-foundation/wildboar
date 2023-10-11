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

from ..distance._cdistance cimport MetricList
from ..utils cimport TSArray
from ..utils._misc cimport to_ndarray_double
from ..utils._rand cimport RAND_R_MAX, RandomSampler, rand_int

from ..distance._distance import _METRICS

from ._attr_gen cimport Attribute, AttributeGenerator


cdef struct TransientPivot:
    Py_ssize_t sample
    Py_ssize_t metric

cdef struct PersitentPivot:
    double *data
    Py_ssize_t length
    Py_ssize_t metric

cdef class PivotAttributeGenerator(AttributeGenerator):
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

    cdef Py_ssize_t get_n_attributess(self, TSArray X) noexcept nogil:
        return self.n_pivots

    cdef Py_ssize_t get_n_outputs(self, TSArray X) noexcept nogil:
        return self.get_n_attributess(X)

    cdef Py_ssize_t next_attribute(
            self,
            Py_ssize_t attribute_id,
            TSArray X,
            Py_ssize_t *samples,
            Py_ssize_t n_samples,
            Attribute *transient,
            uint32_t *seed,
    ) noexcept nogil:
        cdef TransientPivot *pivot = <TransientPivot*> malloc(sizeof(TransientPivot))
        pivot.sample = samples[rand_int(0, n_samples, seed)]
        pivot.metric = self.sampler.rand_int(seed)
        transient.dim = rand_int(0, X.shape[1], seed)
        transient.attribute = pivot
        return 0

    cdef Py_ssize_t free_transient(self, Attribute *attribute) noexcept nogil:
        if attribute.attribute != NULL:
            free(attribute.attribute)
            attribute.attribute = NULL
        return 0

    cdef Py_ssize_t free_persistent(self, Attribute *attribute) noexcept nogil:
        cdef PersitentPivot *pivot
        if attribute.attribute != NULL:
            pivot = <PersitentPivot*> attribute.attribute
            free(pivot.data)
            free(attribute.attribute)
            attribute.attribute = NULL
        return 0

    cdef Py_ssize_t init_persistent(
            self,
            TSArray X,
            Attribute *transient,
            Attribute *persistent
    ) noexcept nogil:
        cdef TransientPivot *pivot = <TransientPivot*> transient.attribute
        cdef PersitentPivot *persistent_pivot = <PersitentPivot*> malloc(
            sizeof(PersitentPivot)
        )

        persistent_pivot.data = <double*> malloc(sizeof(double) * X.shape[2])
        persistent_pivot.metric = pivot.metric
        persistent_pivot.length = X.shape[2]
        memcpy(
            persistent_pivot.data,
            &X[pivot.sample, transient.dim, 0],
            sizeof(double) * X.shape[2]
        )

        persistent.dim = transient.dim
        persistent.attribute = persistent_pivot
        return 0

    cdef double transient_value(
            self,
            Attribute *attribute,
            TSArray X,
            Py_ssize_t sample
    ) noexcept nogil:
        cdef TransientPivot* pivot = <TransientPivot*>attribute.attribute
        return self.metrics.distance(
            pivot.metric, X, sample, X, pivot.sample, attribute.dim
        )

    cdef double persistent_value(
            self,
            Attribute *attribute,
            TSArray X,
            Py_ssize_t sample
    ) noexcept nogil:
        cdef PersitentPivot* pivot = <PersitentPivot*> attribute.attribute
        return self.metrics._distance(
            pivot.metric,
            &X[sample, attribute.dim, 0],
            X.shape[2],
            pivot.data,
            X.shape[2],
        )

    cdef Py_ssize_t transient_fill(
            self,
            Attribute *attribute,
            TSArray X,
            Py_ssize_t sample,
            double[:, :] out,
            Py_ssize_t out_sample,
            Py_ssize_t out_attribute,
    ) noexcept nogil:
        out[out_sample, out_attribute] = self.transient_value(attribute, X, sample)
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
        out[out_sample, out_attribute] = self.persistent_value(attribute, X, sample)
        return 0

    cdef object persistent_to_object(self, Attribute *attribute):
        cdef PersitentPivot *pivot = <PersitentPivot*> attribute.attribute
        return attribute.dim, (
            pivot.metric, to_ndarray_double(pivot.data, pivot.length)
        )

    cdef Py_ssize_t persistent_from_object(self, object object, Attribute *attribute):
        dim, (metric, arr) = object
        cdef PersitentPivot *pivot = <PersitentPivot*> malloc(sizeof(PersitentPivot))
        cdef double *data = <double*> malloc(sizeof(double) * arr.size)
        cdef Py_ssize_t i
        for i in range(arr.size):
            data[i] = arr[i]
        pivot.data = data
        pivot.length = arr.size
        pivot.metric = metric

        attribute.dim = dim
        attribute.attribute = pivot
        return 0
