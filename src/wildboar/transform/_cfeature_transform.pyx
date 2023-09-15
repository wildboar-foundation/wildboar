# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: initializedcheck=False

# Authors: Isak Samsten
# License: BSD 3 clause
from libc.stdlib cimport free, malloc
from numpy cimport uint32_t

from ..utils cimport TSArray
from ..utils._misc cimport safe_realloc
from ..utils._rand cimport RAND_R_MAX
from ._feature cimport Feature, FeatureEngineer

from copy import deepcopy

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs


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
    cdef double[:, :] x_out
    cdef TSArray x_in

    def __init__(self, list feature_engineers, FeatureEmbedding embedding):
        self.feature_engineers = feature_engineers
        self.embedding = embedding

    cdef void init(self, TSArray x_in, double[:, :] x_out):
        self.x_in = x_in
        self.x_out = x_out

cdef class BatchTransform(Batch):

    def __call__(self, Py_ssize_t job_id, Py_ssize_t feature_offset, Py_ssize_t batch_size):
        cdef Py_ssize_t i, j
        cdef FeatureEngineer feature_engineer = self.feature_engineers[job_id]
        feature_engineer.reset(self.x_in)
        with nogil:
            for i in range(self.x_in.shape[0]):
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

    def __call__(self, Py_ssize_t job_id, Py_ssize_t feature_offset, Py_ssize_t batch_size, uint32_t seed):
        cdef Py_ssize_t i, j
        cdef FeatureEngineer feature_engineer = self.feature_engineers[job_id]
        feature_engineer.reset(self.x_in)
        cdef Py_ssize_t *samples = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * self.x_in.shape[0])
        cdef Feature transient_feature
        cdef Feature *persistent_feature

        with nogil:
            for i in range(self.x_in.shape[0]):
                samples[i] = i

            for j in range(batch_size):
                feature_engineer.next_feature(
                    feature_offset + j,
                    self.x_in,
                    samples,
                    self.x_in.shape[0],
                    &transient_feature,
                    &seed,
                )
                persistent_feature = <Feature*> malloc(sizeof(Feature))
                feature_engineer.init_persistent_feature(
                    self.x_in, &transient_feature, persistent_feature
                )
                self.embedding.set_feature(feature_offset + j, persistent_feature)
                for i in range(self.x_in.shape[0]):
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

    cdef Py_ssize_t set_feature(self, Py_ssize_t i, Feature *feature) noexcept nogil:
        """Add a feature to the embedding

        feature : Feature*
            The feature.

            Note: The feature must be allocated using malloc
        """
        self._features[i] = feature
        return 0

    cdef Feature* get_feature(self, Py_ssize_t i) noexcept nogil:
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

def feature_transform_fit(FeatureEngineer feature_engineer, TSArray X, object random_state):
    cdef Py_ssize_t i
    cdef Feature transient_feature
    cdef Feature *persistent_feature
    cdef Py_ssize_t *samples = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * X.shape[0])
    feature_engineer.reset(X)
    cdef FeatureEmbedding embedding = FeatureEmbedding(
        feature_engineer, feature_engineer.get_n_features(X),
    )
    cdef uint32_t seed = random_state.randint(0, RAND_R_MAX)
    
    with nogil:
        for i in range(X.shape[0]):
            samples[i] = i
        
        for i in range(feature_engineer.get_n_features(X)):
            persistent_feature = <Feature*> malloc(sizeof(Feature))
            feature_engineer.next_feature(
                i,
                X,
                samples,
                X.shape[0],
                &transient_feature,
                &seed,
            )
            feature_engineer.init_persistent_feature(
                X, &transient_feature, persistent_feature
            )
            feature_engineer.free_transient_feature(&transient_feature)
            embedding.set_feature(i, persistent_feature)

    return embedding

def feature_transform_transform(FeatureEmbedding embedding, TSArray X, n_jobs=None):
    cdef FeatureEngineer feature_engineer = embedding.feature_engineer
    cdef Py_ssize_t n_outputs = feature_engineer.get_n_outputs(X)
    cdef Py_ssize_t n_features = feature_engineer.get_n_features(X)
    cdef double[:, :] out = np.empty((X.shape[0], n_outputs))

    n_jobs, feature_engineers, feature_offsets, batch_sizes = _partition_features(
        n_jobs, n_features, feature_engineer
    )
    cdef BatchTransform transform = BatchTransform(feature_engineers, embedding)
    transform.init(X, out)

    Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(transform)(
            jobid, feature_offsets[jobid], batch_sizes[jobid]
        )
        for jobid in range(n_jobs)
    )
    
    return out.base

def feature_transform_fit_transform(FeatureEngineer feature_engineer, TSArray X, random_state, n_jobs=None):
    cdef Py_ssize_t n_outputs = feature_engineer.get_n_outputs(X)
    cdef Py_ssize_t n_features = feature_engineer.get_n_features(X)
    cdef FeatureEmbedding embedding = FeatureEmbedding(feature_engineer, n_features)
    cdef double[:, :] out = np.empty((X.shape[0], n_outputs))

    n_jobs, feature_engineers, feature_offsets, batch_sizes = _partition_features(
        n_jobs, n_features, feature_engineer
    )
    seeds = random_state.randint(0, RAND_R_MAX, size=n_jobs)

    cdef BatchFitTransform fit_transform = BatchFitTransform(feature_engineers, embedding)
    fit_transform.init(X, out)
    Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(fit_transform)(
            jobid, feature_offsets[jobid], batch_sizes[jobid], seeds[jobid]
        )
        for jobid in range(n_jobs)
    )

    return embedding, out.base
