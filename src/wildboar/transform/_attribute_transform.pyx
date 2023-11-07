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
from ._attr_gen cimport Attribute, AttributeGenerator

from copy import deepcopy

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs


def clone_embedding(AttributeGenerator generator, attributes):
    cdef AttributeEmbedding embedding = AttributeEmbedding(generator, len(attributes))
    cdef Py_ssize_t i
    cdef Attribute *attribute
    for i in range(len(attributes)):
        attribute = <Attribute*> malloc(sizeof(Attribute))
        generator.persistent_from_object(attributes[i], attribute)
        embedding.set_attribute(i, attribute)
    return embedding


def _partition_attributes(n_jobs, n_attributes, generator):
    n_jobs = min(effective_n_jobs(n_jobs), n_attributes)

    batch_size = n_attributes // n_jobs
    overflow = n_attributes % n_jobs
    generators = []
    attribute_offsets = []
    current_offset = 0
    batch_sizes = []
    for i in range(n_jobs):
        generators.append(deepcopy(generator))
        current_overflow = 0
        if i < overflow:
            current_overflow = 1
        current_batch_size = batch_size + current_overflow
        attribute_offsets.append(current_offset)
        batch_sizes.append(current_batch_size)
        current_offset += current_batch_size

    return n_jobs, generators, attribute_offsets, batch_sizes


cdef class Batch:

    cdef list generators
    cdef AttributeEmbedding embedding
    cdef double[:, :] x_out
    cdef TSArray x_in

    def __init__(self, list generators, AttributeEmbedding embedding):
        self.generators = generators
        self.embedding = embedding

    cdef void init(self, TSArray x_in, double[:, :] x_out):
        self.x_in = x_in
        self.x_out = x_out

cdef class BatchTransform(Batch):

    def __call__(self, Py_ssize_t job_id, Py_ssize_t attribute_offset, Py_ssize_t batch_size):
        cdef Py_ssize_t i, j
        cdef AttributeGenerator generator = self.generators[job_id]
        generator.reset(self.x_in)
        with nogil:
            for i in range(self.x_in.shape[0]):
                for j in range(batch_size):
                    generator.persistent_fill(
                        self.embedding.get_attribute(attribute_offset + j),
                        self.x_in,
                        i,
                        self.x_out,
                        i,
                        attribute_offset + j,
                    )

cdef class BatchFitTransform(Batch):

    def __call__(self, Py_ssize_t job_id, Py_ssize_t attribute_offset, Py_ssize_t batch_size, uint32_t seed):
        cdef Py_ssize_t i, j
        cdef AttributeGenerator generator = self.generators[job_id]
        generator.reset(self.x_in)
        cdef Py_ssize_t *samples = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * self.x_in.shape[0])
        cdef Attribute transient
        cdef Attribute *persistent

        with nogil:
            for i in range(self.x_in.shape[0]):
                samples[i] = i

            for j in range(batch_size):
                generator.next_attribute(
                    attribute_offset + j,
                    self.x_in,
                    samples,
                    self.x_in.shape[0],
                    &transient,
                    &seed,
                )
                for i in range(self.x_in.shape[0]):
                    generator.transient_fill(
                        &transient,
                        self.x_in,
                        i,
                        self.x_out,
                        i,
                        attribute_offset + j,
                    )
                persistent = <Attribute*> malloc(sizeof(Attribute))
                generator.init_persistent(
                    self.x_in, &transient, persistent
                )
                self.embedding.set_attribute(attribute_offset + j, persistent)
                generator.free_transient(&transient)

cdef class AttributeEmbedding:

    cdef AttributeGenerator generator
    cdef Attribute** _attributes
    cdef Py_ssize_t _n_attributes

    def __cinit__(self, AttributeGenerator generator, Py_ssize_t n_attributes):
        self.generator = generator
        self._attributes = <Attribute**> malloc(sizeof(Attribute*) * n_attributes)
        self._n_attributes = n_attributes

    def __dealloc__(self):
        cdef Py_ssize_t i
        for i in range(self._n_attributes):
            self.generator.free_persistent(self._attributes[i])
            free(self._attributes[i])
        free(self._attributes)

    def __reduce__(self):
        return clone_embedding, (
            self.generator,
            self.attributes,
        )

    cdef Py_ssize_t set_attribute(self, Py_ssize_t i, Attribute *attribute) noexcept nogil:
        """Add a attribute to the embedding

        attribute : Attribute*
            The attribute.

            Note: The attribute must be allocated using malloc
        """
        self._attributes[i] = attribute
        return 0

    cdef Attribute* get_attribute(self, Py_ssize_t i) noexcept nogil:
        return self._attributes[i]

    @property
    def n_features(self):
        import warnings
        warnings.warn(
            "`n_features` has been renamed to `n_attributes` in 1.2 and will be removed in 1.4",
            DeprecationWarning
        )
        return self.n_attributes

    @property
    def n_attributes(self):
        return self._n_attributes

    @property
    def features(self):
        import warnings
        warnings.warn(
            "`n_features` has been renamed to `n_attributes` in 1.2 and will be removed in 1.4",
            DeprecationWarning
        )
        return self.attributes

    @property
    def attributes(self):
        return [
            self.generator.persistent_to_object(self._attributes[i])
            for i in range(self._n_attributes)
        ]

    def __getitem__(self, item):
        if not isinstance(item, int) or 0 > item > self._n_attributes:
            raise ValueError()
        return self.generator.persistent_to_object(self._attributes[item])


def fit(AttributeGenerator generator, TSArray X, object random_state):
    cdef Py_ssize_t i
    cdef Attribute transient
    cdef Attribute *persistent
    cdef Py_ssize_t *samples = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * X.shape[0])
    generator.reset(X)
    cdef AttributeEmbedding embedding = AttributeEmbedding(
        generator, generator.get_n_attributess(X),
    )
    cdef uint32_t seed = random_state.randint(0, RAND_R_MAX)

    with nogil:
        for i in range(X.shape[0]):
            samples[i] = i

        for i in range(generator.get_n_attributess(X)):
            persistent = <Attribute*> malloc(sizeof(Attribute))
            generator.next_attribute(
                i,
                X,
                samples,
                X.shape[0],
                &transient,
                &seed,
            )
            generator.init_persistent(
                X, &transient, persistent
            )
            generator.free_transient(&transient)
            embedding.set_attribute(i, persistent)

    return embedding


def transform(AttributeEmbedding embedding, TSArray X, n_jobs=None):
    cdef AttributeGenerator generator = embedding.generator
    cdef Py_ssize_t n_outputs = generator.get_n_outputs(X)
    cdef Py_ssize_t n_attributes = generator.get_n_attributess(X)
    cdef double[:, :] out = np.empty((X.shape[0], n_outputs))

    n_jobs, generators, attribute_offsets, batch_sizes = _partition_attributes(
        n_jobs, n_attributes, generator
    )
    cdef BatchTransform transform = BatchTransform(generators, embedding)
    transform.init(X, out)

    Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(transform)(
            jobid, attribute_offsets[jobid], batch_sizes[jobid]
        )
        for jobid in range(n_jobs)
    )

    return out.base


def fit_transform(AttributeGenerator generator, TSArray X, random_state, n_jobs=None):
    cdef Py_ssize_t n_outputs = generator.get_n_outputs(X)
    cdef Py_ssize_t n_attributes = generator.get_n_attributess(X)
    cdef AttributeEmbedding embedding = AttributeEmbedding(generator, n_attributes)
    cdef double[:, :] out = np.empty((X.shape[0], n_outputs))

    n_jobs, generators, attribute_offsets, batch_sizes = _partition_attributes(
        n_jobs, n_attributes, generator
    )
    seeds = random_state.randint(0, RAND_R_MAX, size=n_jobs)

    cdef BatchFitTransform fit_transform = BatchFitTransform(generators, embedding)
    fit_transform.init(X, out)
    Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(fit_transform)(
            jobid, attribute_offsets[jobid], batch_sizes[jobid], seeds[jobid]
        )
        for jobid in range(n_jobs)
    )

    return embedding, out.base


def derivative_transform(TSArray X):
    if X.shape[2] < 3:
        return X.base

    cdef Py_ssize_t i, j, k
    cdef double[:, :, :] out = np.empty(
        (X.shape[0], X.shape[1], X.shape[2] - 2), dtype=float
    )

    for i in range(X.shape[0]):
        for k in range(X.shape[1]):
            for j in range(1, X.shape[2] - 1):
                out[i, k, j - 1] = (
                    (X[i, k, j] - X[i, k, j - 1])
                    + ((X[i, k, j + 1] - X[i, k, j - 1]) / 2)
                ) / 2

    return out.base
