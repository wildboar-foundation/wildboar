# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: initializedcheck=False

# Authors: Isak Samsten
# License: BSD 3 clause

from libc.math cimport NAN
from libc.stdlib cimport free, malloc
from numpy cimport uint32_t

from ..utils cimport TSArray


cdef class FeatureEngineer:

    cdef int reset(self, TSArray X) noexcept nogil:
        return 0

    cdef Py_ssize_t get_n_features(self, TSArray X) noexcept nogil:
        return -1

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
        return -1

    cdef Py_ssize_t free_transient_feature(self, Feature *feature) noexcept nogil:
        return -1

    cdef Py_ssize_t free_persistent_feature(self, Feature *feature) noexcept nogil:
        return -1

    cdef Py_ssize_t init_persistent_feature(
        self, 
        TSArray X,
        Feature *transient, 
        Feature *persistent
    ) noexcept nogil:
        return 0

    cdef double transient_feature_value(
        self,
        Feature *feature,
        TSArray X,
        Py_ssize_t sample
    ) noexcept nogil:
        return NAN

    cdef double persistent_feature_value(
        self,
        Feature *feature,
        TSArray X,
        Py_ssize_t sample
    ) noexcept nogil:
        return NAN

    cdef Py_ssize_t transient_feature_fill(
        self, 
        Feature *feature, 
        TSArray X, 
        Py_ssize_t sample,
        double[:, :] out,
        Py_ssize_t out_sample,
        Py_ssize_t out_feature,
    ) noexcept nogil:
        return -1

    cdef Py_ssize_t persistent_feature_fill(
        self, 
        Feature *feature, 
        TSArray X, 
        Py_ssize_t sample,
        double[:, :] out,
        Py_ssize_t out_sample,
        Py_ssize_t out_feature,
    ) noexcept nogil:
        return -1

    cdef object persistent_feature_to_object(self, Feature *feature):
        return None

    cdef Py_ssize_t persistent_feature_from_object(self, object object, Feature *feature):
        return 0

    cdef void transient_feature_values(
        self, 
        Feature *feature, 
        TSArray X, 
        Py_ssize_t *samples, 
        Py_ssize_t n_samples,
        double* values
    ) noexcept nogil:
        cdef Py_ssize_t i
        for i in range(n_samples):
            values[i] = self.transient_feature_value(feature, X, samples[i])
    
    cdef void persistent_feature_values(
        self, 
        Feature *feature, 
        TSArray X, 
        Py_ssize_t *samples, 
        Py_ssize_t n_samples,
        double* values
    ) noexcept nogil:
        cdef Py_ssize_t i
        for i in range(n_samples):
            values[i] = self.persistent_feature_value(feature, X, samples[i])
