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
cimport numpy as np
from libc.stdlib cimport free, malloc

from .._data cimport TSDatabase, ts_database_new
from .._utils cimport RAND_R_MAX, safe_realloc
from ._feature cimport Feature, FeatureEngineer

from copy import deepcopy

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs

from .._data import ts_database_dims


def clone_embedding(FeatureEngineer feature_engineer, features):
    cdef FeatureEmbedding embedding = FeatureEmbedding(feature_engineer, len(features))
    cdef Py_ssize_t i
    cdef Feature *feature
    for i in range(len(features)):
        feature = <Feature*> malloc(sizeof(Feature))
        feature_engineer.persistent_feature_from_object(features[i], feature)
        embedding.set_feature(i, feature)
    return embedding

def _partition_features(n_jobs, n_features, feature_engineer):
    n_jobs = min(effective_n_jobs(n_jobs), n_features)

    batch_size = n_features // n_jobs
    overflow = n_features % n_jobs
    feature_engineers = []
    feature_offsets = []
    current_offset = 0
    batch_sizes = []
    for i in range(n_jobs):
        feature_engineers.append(deepcopy(feature_engineer))
        current_overflow = 0
        if i < overflow:
            current_overflow = 1
        current_batch_size = batch_size + current_overflow
        feature_offsets.append(current_offset)
        batch_sizes.append(current_batch_size)
        current_offset += current_batch_size
    
    return n_jobs, feature_engineers, feature_offsets, batch_sizes


cdef class Batch:

    cdef list feature_engineers
    cdef FeatureEmbedding embedding
    cdef TSDatabase *x_out
    cdef TSDatabase *x_in

    def __init__(self, list feature_engineers, FeatureEmbedding embedding):
        self.feature_engineers = feature_engineers
        self.embedding = embedding

    cdef void init(self, TSDatabase *x_in, TSDatabase *x_out):
        self.x_in = x_in
        self.x_out = x_out

cdef class BatchTransform(Batch):

    def __call__(self, Py_ssize_t job_id, Py_ssize_t feature_offset, Py_ssize_t batch_size):
        cdef Py_ssize_t i, j
        cdef FeatureEngineer feature_engineer = self.feature_engineers[job_id]
        with nogil:
            for i in range(self.x_in.n_samples):
                for j in range(batch_size):
                    feature_engineer.persistent_feature_fill(
                        self.embedding.get_feature(feature_offset + j),
                        self.x_in,
                        i,
                        self.x_out,
                        i,
                        feature_offset + j,
                    )

cdef class BatchFitTransform(Batch):

    def __call__(self, Py_ssize_t job_id, Py_ssize_t feature_offset, Py_ssize_t batch_size, size_t seed):
        cdef Py_ssize_t i, j
        cdef FeatureEngineer feature_engineer = self.feature_engineers[job_id]
        cdef Py_ssize_t *samples = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * self.x_in.n_samples)
        cdef Feature transient_feature
        cdef Feature *persistent_feature

        with nogil:
            for i in range(self.x_in.n_samples):
                samples[i] = i

            for j in range(batch_size):
                feature_engineer.next_feature(
                    feature_offset + j,
                    self.x_in,
                    samples,
                    self.x_in.n_samples,
                    &transient_feature,
                    &seed,
                )
                persistent_feature = <Feature*> malloc(sizeof(Feature))
                feature_engineer.init_persistent_feature(
                    self.x_in, &transient_feature, persistent_feature
                )
                self.embedding.set_feature(feature_offset + j, persistent_feature)
                for i in range(self.x_in.n_samples):
                    feature_engineer.transient_feature_fill(
                        &transient_feature,
                        self.x_in,
                        i,
                        self.x_out,
                        i,
                        feature_offset + j,
                    )

                feature_engineer.free_transient_feature(&transient_feature)

cdef class FeatureEmbedding:

    cdef FeatureEngineer feature_engineer
    cdef Feature** _features
    cdef Py_ssize_t _n_features

    def __cinit__(self, FeatureEngineer feature_engineer, Py_ssize_t n_features):
        self.feature_engineer = feature_engineer
        self._features = <Feature**> malloc(sizeof(Feature*) * n_features)
        self._n_features = n_features

    def __dealloc__(self):
        cdef Py_ssize_t i
        for i in range(self._n_features):
            self.feature_engineer.free_persistent_feature(self._features[i])
            free(self._features[i])
        free(self._features)

    def __reduce__(self):
        return clone_embedding, (
            self.feature_engineer, 
            self.features,
        )

    cdef Py_ssize_t set_feature(self, Py_ssize_t i, Feature *feature) nogil:
        """Add a feature to the embedding

        feature : Feature*
            The feature.

            Note: The feature must be allocated using malloc
        """
        self._features[i] = feature
        return 0

    cdef Feature* get_feature(self, Py_ssize_t i) nogil:
        return self._features[i]

    @property
    def n_features(self):
        return self._n_features
    
    @property
    def features(self):
        return [
            self.feature_engineer.persistent_feature_to_object(self._features[i]) 
            for i in range(self._n_features)
        ]

    def __getitem__(self, item):
        if not isinstance(item, int) or 0 > item > self._n_features:
            raise ValueError()
        return self.feature_engineer.persistent_feature_to_object(self._features[item])

def feature_embedding_fit(FeatureEngineer feature_engineer, np.ndarray x, object random_state):
    cdef TSDatabase td = ts_database_new(x)
    cdef Py_ssize_t i
    cdef Feature transient_feature
    cdef Feature *persistent_feature
    cdef Py_ssize_t *samples = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * td.n_samples)
    cdef FeatureEmbedding embedding = FeatureEmbedding(
        feature_engineer, feature_engineer.get_n_features(&td),
    )
    cdef size_t seed = random_state.randint(0, RAND_R_MAX)
    
    with nogil:
        for i in range(td.n_samples):
            samples[i] = i
        
        for i in range(feature_engineer.get_n_features(&td)):
            persistent_feature = <Feature*> malloc(sizeof(Feature))
            feature_engineer.next_feature(
                i,
                &td,
                samples,
                td.n_samples,
                &transient_feature,
                &seed,
            )
            feature_engineer.init_persistent_feature(
                &td, &transient_feature, persistent_feature
            )
            feature_engineer.free_transient_feature(&transient_feature)
            embedding.set_feature(i, persistent_feature)

    return embedding

def feature_embedding_transform(FeatureEmbedding embedding, np.ndarray x, n_jobs=None):
    cdef TSDatabase x_in = ts_database_new(x)
    cdef FeatureEngineer feature_engineer = embedding.feature_engineer
    cdef Py_ssize_t n_outputs = feature_engineer.get_n_outputs(&x_in)
    cdef Py_ssize_t n_features = feature_engineer.get_n_features(&x_in)
    cdef np.ndarray out = np.empty((x.shape[0], n_outputs))
    cdef TSDatabase x_out = ts_database_new(out)

    n_jobs, feature_engineers, feature_offsets, batch_sizes = _partition_features(
        n_jobs, n_features, feature_engineer
    )
    cdef BatchTransform transform = BatchTransform(feature_engineers, embedding)
    transform.init(&x_in, &x_out)

    Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(transform)(
            jobid, feature_offsets[jobid], batch_sizes[jobid]
        )
        for jobid in range(n_jobs)
    )
    
    return out

def feature_embedding_fit_transform(FeatureEngineer feature_engineer, np.ndarray x, random_state, n_jobs=None):
    cdef TSDatabase x_in = ts_database_new(x)
    cdef Py_ssize_t n_outputs = feature_engineer.get_n_outputs(&x_in)
    cdef Py_ssize_t n_features = feature_engineer.get_n_features(&x_in)
    cdef FeatureEmbedding embedding = FeatureEmbedding(
        feature_engineer, n_features
    )
    cdef np.ndarray out = np.empty((x.shape[0], n_outputs))
    cdef TSDatabase x_out = ts_database_new(out)

    n_jobs, feature_engineers, feature_offsets, batch_sizes = _partition_features(
        n_jobs, n_features, feature_engineer
    )
    seeds = random_state.randint(0, RAND_R_MAX, size=n_jobs)

    cdef BatchFitTransform fit_transform = BatchFitTransform(feature_engineers, embedding)
    fit_transform.init(&x_in, &x_out)
    Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(fit_transform)(
            jobid, feature_offsets[jobid], batch_sizes[jobid], seeds[jobid]
        )
        for jobid in range(n_jobs)
    )

    return embedding, out