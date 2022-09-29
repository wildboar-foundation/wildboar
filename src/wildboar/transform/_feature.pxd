# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

from wildboar.utils.data cimport Dataset


cdef struct Feature:
    Py_ssize_t dim       # the dimension of the feature -1 for undefined
    void* feature        # the feature value
    

cdef class FeatureEngineer:

    cdef Py_ssize_t reset(self, Dataset td) nogil

    cdef Py_ssize_t get_n_features(self, Dataset td) nogil

    cdef Py_ssize_t get_n_outputs(self, Dataset td) nogil

    cdef Py_ssize_t next_feature(
        self,
        Py_ssize_t feature_id,
        Dataset td, 
        Py_ssize_t *samples, 
        Py_ssize_t n_samples,
        Feature *transient,
        size_t *seed
    ) nogil

    # Initialize a persisent feature from a transient feature
    cdef Py_ssize_t init_persistent_feature(
        self, 
        Dataset td,
        Feature *transient, 
        Feature *persistent
    ) nogil
    
    cdef Py_ssize_t free_transient_feature(self, Feature *feature) nogil

    cdef Py_ssize_t free_persistent_feature(self, Feature *feature) nogil

    # Calculate the feature value for sample using the transient feature
    cdef double transient_feature_value(
        self,
        Feature *feature,
        Dataset td,
        Py_ssize_t sample
    ) nogil

    cdef Py_ssize_t transient_feature_fill(
        self, 
        Feature *feature, 
        Dataset td, 
        Py_ssize_t sample,
        Dataset td_out,
        Py_ssize_t out_sample,
        Py_ssize_t out_feature,
    ) nogil

    # Calculate the feature value for sample using the persistent feature
    cdef double persistent_feature_value(
        self,
        Feature *feature,
        Dataset td,
        Py_ssize_t sample
    ) nogil

    cdef Py_ssize_t persistent_feature_fill(
        self, 
        Feature *feature, 
        Dataset td, 
        Py_ssize_t sample,
         Dataset td_out,
        Py_ssize_t out_sample,
        Py_ssize_t out_feature,
    ) nogil
    
    # Calculate the feature value for all samples using the transient featuer
    cdef void transient_feature_values(
        self, 
        Feature *feature, 
        Dataset td, 
        Py_ssize_t *samples, 
        Py_ssize_t n_samples,
        double* values
    ) nogil
    
    # Calculate the feature value for all samples using the persistent feature
    cdef void persistent_feature_values(
        self, 
        Feature *feature, 
        Dataset td, 
        Py_ssize_t *samples, 
        Py_ssize_t n_samples,
        double* values
    ) nogil

    cdef object persistent_feature_to_object(self, Feature *feature)

    cdef Py_ssize_t persistent_feature_from_object(self, object object, Feature *feature)