# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

cimport numpy as np

import numpy as np

from libc.math cimport NAN
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy

from ..distance._distance cimport DistanceMeasure
from ..utils.data cimport Dataset
from ..utils.misc cimport CList, to_ndarray_double
from ..utils.rand cimport RAND_R_MAX, rand_int, shuffle

from ..distance import _DISTANCE_MEASURE

from ._feature cimport Feature, FeatureEngineer


cdef struct TransientPivot:
    Py_ssize_t sample
    Py_ssize_t distance_measure

cdef struct PersitentPivot:
    double *data
    Py_ssize_t length
    Py_ssize_t distance_measure

cdef class PivotFeatureEngineer(FeatureEngineer):
    cdef Py_ssize_t n_pivots
    cdef CList distance_measures

    def __cinit__(self, Py_ssize_t n_pivots, list distance_measures):
        self.n_pivots = n_pivots
        self.distance_measures = CList(distance_measures)

    def __reduce__(self):
        return self.__class__, (self.n_pivots, self.distance_measures.py_list)

    cdef Py_ssize_t reset(self, Dataset td) nogil:
        cdef Py_ssize_t i
        for i in range(self.distance_measures.size):
            (<DistanceMeasure>self.distance_measures.get(i)).reset(td, td)
        return 0

    cdef Py_ssize_t get_n_features(self, Dataset td) nogil:
        return self.n_pivots

    cdef Py_ssize_t get_n_outputs(self, Dataset td) nogil:
        return self.get_n_features(td)

    cdef Py_ssize_t next_feature(
            self,
            Py_ssize_t feature_id,
            Dataset td,
            Py_ssize_t *samples,
            Py_ssize_t n_samples,
            Feature *transient,
            size_t *seed,
    ) nogil:
        cdef TransientPivot *pivot = <TransientPivot*> malloc(sizeof(TransientPivot))
        pivot.sample = samples[rand_int(0, n_samples, seed)]
        pivot.distance_measure = rand_int(0, self.distance_measures.size, seed)
        transient.dim = rand_int(0, td.n_dims, seed)
        transient.feature = pivot
        return 0

    cdef Py_ssize_t free_transient_feature(self, Feature *feature) nogil:
        if feature.feature != NULL:
            free(feature.feature)
            feature.feature = NULL
        return 0

    cdef Py_ssize_t free_persistent_feature(self, Feature *feature) nogil:
        cdef PersitentPivot *pivot
        if feature.feature != NULL:
            pivot = <PersitentPivot*> feature.feature
            free(pivot.data)
            free(feature.feature)
            feature.feature = NULL
        return 0

    cdef Py_ssize_t init_persistent_feature(
            self,
            Dataset td,
            Feature *transient,
            Feature *persistent
    ) nogil:
        cdef TransientPivot *pivot = <TransientPivot*> transient.feature
        cdef PersitentPivot *persistent_pivot = <PersitentPivot*> malloc(sizeof(PersitentPivot))
        
        persistent_pivot.data = <double*> malloc(sizeof(double) * td.n_timestep)
        persistent_pivot.distance_measure = pivot.distance_measure
        persistent_pivot.length = td.n_timestep
        memcpy(
            persistent_pivot.data, 
            td.get_sample(pivot.sample, transient.dim), 
            sizeof(double) * td.n_timestep
        )
        
        persistent.dim = transient.dim
        persistent.feature = persistent_pivot
        return 0

    cdef double transient_feature_value(
            self,
            Feature *feature,
            Dataset td,
            Py_ssize_t sample
    ) nogil:
        cdef TransientPivot* pivot = <TransientPivot*>feature.feature
        return (<DistanceMeasure>self.distance_measures.get(pivot.distance_measure)).distance(
            td, sample, td, pivot.sample, feature.dim
        )

    cdef double persistent_feature_value(
            self,
            Feature *feature,
            Dataset td,
            Py_ssize_t sample
    ) nogil:
        cdef PersitentPivot* pivot = <PersitentPivot*> feature.feature
        return (<DistanceMeasure>self.distance_measures.get(pivot.distance_measure))._distance(
            td.get_sample(sample, feature.dim),
            td.n_timestep,
            pivot.data,
            td.n_timestep,
        )

    cdef Py_ssize_t transient_feature_fill(
            self,
            Feature *feature,
            Dataset td,
            Py_ssize_t sample,
            Dataset td_out,
            Py_ssize_t out_sample,
            Py_ssize_t out_feature,
    ) nogil:
        td_out.get_sample(out_sample, feature.dim)[out_feature] = self.transient_feature_value(
            feature, td, sample
        )
        return 0

    cdef Py_ssize_t persistent_feature_fill(
            self,
            Feature *feature,
            Dataset td,
            Py_ssize_t sample,
            Dataset td_out,
            Py_ssize_t out_sample,
            Py_ssize_t out_feature,
    ) nogil:
        td_out.get_sample(out_sample, feature.dim)[out_feature] = self.persistent_feature_value(
            feature, td, sample
        )
        return 0

    cdef object persistent_feature_to_object(self, Feature *feature):
        cdef PersitentPivot *pivot = <PersitentPivot*> feature.feature
        return feature.dim, (pivot.distance_measure, to_ndarray_double(pivot.data, pivot.length))

    cdef Py_ssize_t persistent_feature_from_object(self, object object, Feature *feature):
        dim, (distance_measure, arr) = object
        cdef PersitentPivot *pivot = <PersitentPivot*> malloc(sizeof(PersitentPivot))
        cdef double *data = <double*> malloc(sizeof(double) * arr.size)
        cdef Py_ssize_t i
        for i in range(arr.size):
            data[i] = arr[i]
        pivot.data = data
        pivot.length = arr.size
        pivot.distance_measure = distance_measure

        feature.dim = dim
        feature.feature = pivot
        return 0