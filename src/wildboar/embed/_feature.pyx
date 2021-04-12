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
from libc.math cimport NAN
from libc.stdlib cimport free, malloc

from .._data cimport TSDatabase


cdef class FeatureEngineer:

    cdef Py_ssize_t init(self, TSDatabase *td) nogil:
        return 0

    cdef Py_ssize_t get_n_features(self, TSDatabase *td) nogil:
        return -1

    cdef Py_ssize_t get_n_outputs(self, TSDatabase *td) nogil:
        return self.get_n_features(td)

    cdef Py_ssize_t next_feature(
        self,
        Py_ssize_t feature_id,
        TSDatabase *td, 
        Py_ssize_t *samples, 
        Py_ssize_t n_samples,
        Feature *transient,
        size_t *seed,
    ) nogil:
        return -1

    cdef Py_ssize_t free_transient_feature(self, Feature *feature) nogil:
        return -1

    cdef Py_ssize_t free_persistent_feature(self, Feature *feature) nogil:
        return -1

    cdef Py_ssize_t init_persistent_feature(
        self, 
        TSDatabase *td,
        Feature *transient, 
        Feature *persistent
    ) nogil:
        return 0

    cdef double transient_feature_value(
        self,
        Feature *feature,
        TSDatabase *td,
        Py_ssize_t sample
    ) nogil:
        return NAN

    cdef double persistent_feature_value(
        self,
        Feature *feature,
        TSDatabase *td,
        Py_ssize_t sample
    ) nogil:
        return NAN

    cdef Py_ssize_t transient_feature_fill(
        self, 
        Feature *feature, 
        TSDatabase *td, 
        Py_ssize_t sample,
        TSDatabase *td_out,
        Py_ssize_t out_sample,
        Py_ssize_t out_feature,
    ) nogil:
        return -1

    cdef Py_ssize_t persistent_feature_fill(
        self, 
        Feature *feature, 
        TSDatabase *td, 
        Py_ssize_t sample,
        TSDatabase *td_out,
        Py_ssize_t out_sample,
        Py_ssize_t out_feature,
    ) nogil:
        return -1

    cdef object persistent_feature_to_object(self, Feature *feature):
        return None

    cdef Py_ssize_t persistent_feature_from_object(self, object object, Feature *feature):
        return 0

    cdef void transient_feature_values(
        self, 
        Feature *feature, 
        TSDatabase *td, 
        Py_ssize_t *samples, 
        Py_ssize_t n_samples,
        double* values
    ) nogil:
        cdef Py_ssize_t i
        for i in range(n_samples):
            values[i] = self.transient_feature_value(feature, td, samples[i])
    
    cdef void persistent_feature_values(
        self, 
        Feature *feature, 
        TSDatabase *td, 
        Py_ssize_t *samples, 
        Py_ssize_t n_samples,
        double* values
    ) nogil:
        cdef Py_ssize_t i
        for i in range(n_samples):
            values[i] = self.persistent_feature_value(feature, td, samples[i])