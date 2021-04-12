# cython: language_level=3

# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wildboar is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
# General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Authors: Isak Samsten

from .._data cimport TSDatabase


cdef struct Feature:
    Py_ssize_t dim       # the dimension of the feature -1 for undefined
    void* feature        # the feature value
    

cdef class FeatureEngineer:

    cdef Py_ssize_t init(self, TSDatabase *td) nogil

    cdef Py_ssize_t get_n_features(self, TSDatabase *td) nogil

    cdef Py_ssize_t get_n_outputs(self, TSDatabase *td) nogil

    cdef Py_ssize_t next_feature(
        self,
        Py_ssize_t feature_id,
        TSDatabase *td, 
        Py_ssize_t *samples, 
        Py_ssize_t n_samples,
        Feature *transient,
        size_t *seed
    ) nogil

    # Initialize a persisent feature from a transient feature
    cdef Py_ssize_t init_persistent_feature(
        self, 
        TSDatabase *td,
        Feature *transient, 
        Feature *persistent
    ) nogil
    
    cdef Py_ssize_t free_transient_feature(self, Feature *feature) nogil

    cdef Py_ssize_t free_persistent_feature(self, Feature *feature) nogil

    # Calculate the feature value for sample using the transient feature
    cdef double transient_feature_value(
        self,
        Feature *feature,
        TSDatabase *td,
        Py_ssize_t sample
    ) nogil

    cdef Py_ssize_t transient_feature_fill(
        self, 
        Feature *feature, 
        TSDatabase *td, 
        Py_ssize_t sample,
        TSDatabase *td_out,
        Py_ssize_t out_sample,
        Py_ssize_t out_feature,
    ) nogil

    # Calculate the feature value for sample using the persistent feature
    cdef double persistent_feature_value(
        self,
        Feature *feature,
        TSDatabase *td,
        Py_ssize_t sample
    ) nogil

    cdef Py_ssize_t persistent_feature_fill(
        self, 
        Feature *feature, 
        TSDatabase *td, 
        Py_ssize_t sample,
         TSDatabase *td_out,
        Py_ssize_t out_sample,
        Py_ssize_t out_feature,
    ) nogil
    
    # Calculate the feature value for all samples using the transient featuer
    cdef void transient_feature_values(
        self, 
        Feature *feature, 
        TSDatabase *td, 
        Py_ssize_t *samples, 
        Py_ssize_t n_samples,
        double* values
    ) nogil
    
    # Calculate the feature value for all samples using the persistent feature
    cdef void persistent_feature_values(
        self, 
        Feature *feature, 
        TSDatabase *td, 
        Py_ssize_t *samples, 
        Py_ssize_t n_samples,
        double* values
    ) nogil

    cdef object persistent_feature_to_object(self, Feature *feature)

    cdef Py_ssize_t persistent_feature_from_object(self, object object, Feature *feature)