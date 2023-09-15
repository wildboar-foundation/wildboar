# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

from numpy cimport uint32_t

from wildboar.utils cimport TSArray


cdef struct Feature:
    Py_ssize_t dim       # the dimension of the feature -1 for undefined
    void* feature        # the feature value
    

cdef class FeatureEngineer:

    cdef int reset(self, TSArray X) noexcept nogil

    cdef Py_ssize_t get_n_features(self, TSArray X) noexcept nogil

    cdef Py_ssize_t get_n_outputs(self, TSArray X) noexcept nogil

    cdef Py_ssize_t next_feature(
        self,
        Py_ssize_t feature_id,
        TSArray X, 
        Py_ssize_t *samples, 
        Py_ssize_t n_samples,
        Feature *transient,
        uint32_t *seed
    ) noexcept nogil

    # Initialize a persisent feature from a transient feature
    cdef Py_ssize_t init_persistent_feature(
        self, 
        TSArray X,
        Feature *transient, 
        Feature *persistent
    ) noexcept nogil
    
    cdef Py_ssize_t free_transient_feature(self, Feature *feature) noexcept nogil

    cdef Py_ssize_t free_persistent_feature(self, Feature *feature) noexcept nogil

    # Calculate the feature value for sample using the transient feature
    cdef double transient_feature_value(
        self,
        Feature *feature,
        TSArray X,
        Py_ssize_t sample
    ) noexcept nogil

    cdef Py_ssize_t transient_feature_fill(
        self, 
        Feature *feature, 
        TSArray X, 
        Py_ssize_t sample,
        double[:, :] out,
        Py_ssize_t out_sample,
        Py_ssize_t out_feature,
    ) noexcept nogil

    # Calculate the feature value for sample using the persistent feature
    cdef double persistent_feature_value(
        self,
        Feature *feature,
        TSArray X,
        Py_ssize_t sample
    ) noexcept nogil

    cdef Py_ssize_t persistent_feature_fill(
        self, 
        Feature *feature, 
        TSArray X, 
        Py_ssize_t sample,
        double[:, :] out,
        Py_ssize_t out_sample,
        Py_ssize_t out_feature,
    ) noexcept nogil
    
    # Calculate the feature value for all samples using the transient featuer
    cdef void transient_feature_values(
        self, 
        Feature *feature, 
        TSArray X, 
        Py_ssize_t *samples, 
        Py_ssize_t n_samples,
        double* values
    ) noexcept nogil
    
    # Calculate the feature value for all samples using the persistent feature
    cdef void persistent_feature_values(
        self, 
        Feature *feature, 
        TSArray X, 
        Py_ssize_t *samples, 
        Py_ssize_t n_samples,
        double* values
    ) noexcept nogil

    cdef object persistent_feature_to_object(self, Feature *feature)

    cdef Py_ssize_t persistent_feature_from_object(self, object object, Feature *feature)