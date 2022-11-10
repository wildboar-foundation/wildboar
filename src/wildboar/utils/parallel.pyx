# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause

from abc import ABCMeta, abstractmethod
from copy import deepcopy

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs


cdef class ForeachSample:

    def __cinit__(self, Dataset x, *args, **kwargs):
        self.x_in = x
    
    def __call__(self, Py_ssize_t job_id, Py_ssize_t offset, Py_ssize_t batch_size):
        cdef Py_ssize_t i, dim
        with nogil:
            for i in range(offset, offset + batch_size):
                self.work(i)

    cdef void work(self, Py_ssize_t sample) nogil:
        pass

    @property
    def n_work(self):
        return self.x_in.n_samples


cdef class MapSample(ForeachSample):

    def __cinit__(self, Dataset x, double[:, :] result, *args, **kwargs):
        self.result = result

    cdef void work(self, Py_ssize_t sample) nogil:
        cdef Py_ssize_t dim
        cdef double v
        for dim in range(self.x_in.n_dims):
            self.result[sample, dim] = self.map(self.x_in.get_sample(sample, dim))

    cdef double map(self, double *sample) nogil:
        pass


def partition_n_jobs(n_jobs, n):
    n_jobs = min(effective_n_jobs(n_jobs), n)

    batch_size = n // n_jobs
    overflow = n % n_jobs
    offsets = []
    offset = 0
    batch_sizes = []
    for i in range(n_jobs):
        current_overflow = 0
        if i < overflow:
            current_overflow = 1
        current_batch_size = batch_size + current_overflow
        offsets.append(offset)
        batch_sizes.append(current_batch_size)
        offset += current_batch_size

    return n_jobs, offsets, batch_sizes


def run_in_parallel(work, n_jobs=-1, **parallel_args):
    """Execute work in parallel

    Parameters
    ----------
    work : Batch
        A work batch
    n_jobs : int, optional
        Number of parallel jobs, by default -1
    """
    n_jobs, offsets, batch_sizes = partition_n_jobs(n_jobs, work.n_work)
    Parallel(n_jobs=n_jobs, **parallel_args)(
        delayed(work)(jobid, offsets[jobid], batch_sizes[jobid])
        for jobid in range(n_jobs)
    )
