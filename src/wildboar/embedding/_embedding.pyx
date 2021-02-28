# cython: language_level=3

# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wildboar is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Authors: Isak Samsten
cimport numpy as np
import numpy as np

from libc.stdlib cimport malloc, free

from .._utils cimport safe_realloc

from .._data cimport ts_database_new
from .._data cimport TSDatabase

from ._feature cimport FeatureEngineer
from ._feature cimport Feature

cpdef Embedding clone_embedding(FeatureEngineer feature_engineer, features):
    cdef Embedding embedding = Embedding(feature_engineer, capacity=len(features))
    cdef Py_ssize_t i
    cdef Feature *feature
    for i in range(len(features)):
        feature = <Feature*> malloc(sizeof(Feature))
        feature_engineer.persistent_feature_from_object(features[i], feature)
        embedding.add_feature(feature)
    return embedding


cdef class Embedding:

    cdef FeatureEngineer _feature_engineer
    cdef Feature** _features
    cdef Py_ssize_t _n_features

    def __cinit__(self, FeatureEngineer feature_engineer, Py_ssize_t capacity=100):
        self._feature_engineer = feature_engineer
        self._features = <Feature**> malloc(sizeof(Feature*) * capacity)
        self._n_features = 0

    def __dealloc__(self):
        cdef Py_ssize_t i
        for i in range(self._n_features):
            self._feature_engineer.free_persistent_feature(self._features[i])
            free(self._features[i])
        free(self._features)

    def __reduce__(self):
        return clone_embedding, (
            self._feature_engineer, 
            self.n_features, 
            self.features,
        )

    cdef Py_ssize_t add_feature(self, Feature *feature) nogil:
        """Add a feature to the embedding

        feature : Feature*
            The feature.

            Note: The feature must be allocated using malloc
        """
        cdef Py_ssize_t new_capacity = self._n_features * 2
        cdef Py_ssize_t ret = safe_realloc(<void**> &self._features, sizeof(Feature*) * new_capacity)
        if ret == -1:
            return -1
        self._features[self._n_features] = feature
        self._n_features += 1
        return 0

    @property
    def n_features(self):
        return self._n_features
    
    @property
    def features(self):
        return [
            self._feature_engineer.persistent_feature_to_object(self._features[i]) 
            for i in range(self._n_features)
        ]

    cpdef np.ndarray apply(self, np.ndarray x):
        cdef Py_ssize_t n_outputs
        cdef Py_ssize_t i, j
        cdef Py_ssize_t feature_offset
        cdef np.ndarray out
        cdef TSDatabase td, td_out
        
        out = np.empty((x.shape[0], self._feature_engineer.get_n_outputs(&td)))

        td = ts_database_new(x)
        td_out = ts_database_new(out)
        with nogil:
            for i in range(td.n_samples):
                for j in range(self._n_features):
                    self._feature_engineer.persistent_feature_fill(
                        self._features[j],
                        &td,
                        i,
                        &td_out,
                        i,
                        j,
                    )
        
        return out

cdef class FeatureEngineerEmbedding:

    cdef FeatureEngineer _feature_engineer
    cdef Embedding _embedding

    def __cinit__(self, FeatureEngineer feature_engineer):
        self._feature_engineer = feature_engineer
        self._embedding = None

    @property
    def embedding_(self):
        return self._embedding

    def fit_embedding(self, np.ndarray x):
        cdef TSDatabase td = ts_database_new(x)
        cdef Py_ssize_t i
        cdef Feature transient_feature
        cdef Feature *persistent_feature
        cdef Py_ssize_t *samples = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * td.n_samples)
        cdef Embedding embedding = Embedding(
            self._feature_engineer, 
            capacity=self._feature_engineer.get_n_features(&td),
        )
        
        with nogil:
            for i in range(td.n_samples):
                samples[i] = i
            
            for i in range(self._feature_engineer.get_n_features(&td)):
                persistent_feature = <Feature*> malloc(sizeof(Feature))
                self._feature_engineer.next_feature(
                    i,
                    &td,
                    samples,
                    td.n_samples,
                    &transient_feature
                )
                self._feature_engineer.init_persistent_feature(
                    &td, &transient_feature, persistent_feature
                )
                self._feature_engineer.free_transient_feature(&transient_feature)
                embedding.add_feature(persistent_feature)
        self._embedding = embedding

    def fit_embedding_transform(self, np.ndarray x):
        cdef TSDatabase td = ts_database_new(x)
        cdef Embedding embedding = Embedding(
            self._feature_engineer, 
            capacity=self._feature_engineer.get_n_features(&td)
        )
        cdef Py_ssize_t *samples = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * td.n_samples)
        cdef Py_ssize_t i, j, feature_offset
        cdef Feature transient_feature
        cdef Feature *persistent_feature
        cdef np.ndarray out = np.empty((x.shape[0], self._feature_engineer.get_n_outputs(&td)))
        cdef TSDatabase td_out = ts_database_new(out)
        
        with nogil:
            for i in range(td.n_samples):
                samples[i] = i
            
            for j in range(self._feature_engineer.get_n_features(&td)):
                self._feature_engineer.next_feature(
                    j,
                    &td,
                    samples,
                    td.n_samples,
                    &transient_feature
                )
                persistent_feature = <Feature*> malloc(sizeof(Feature))
                self._feature_engineer.init_persistent_feature(
                    &td, &transient_feature, persistent_feature
                )
                embedding.add_feature(persistent_feature)
                for i in range(td.n_samples):
                    self._feature_engineer.transient_feature_fill(
                        &transient_feature,
                        &td,
                        i,
                        &td_out,
                        i,
                        j,
                    )

                self._feature_engineer.free_transient_feature(&transient_feature)
        self._embedding = embedding
        return out
        




    