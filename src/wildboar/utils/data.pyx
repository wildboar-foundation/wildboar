# cython: language_level=3

# Authors: Isak Samsten
# License: BSD 3 clause
import numpy as np

from cython.view cimport memoryview


cdef extern from "Python.h":
    cdef int PyBUF_STRIDES
    cdef int PyBUF_C_CONTIGUOUS

__all__ = [
    "Dataset",
    "check_dataset",
]

def check_dataset(x, allow_1d=False, allow_2d=True, allow_3d=True):
    """Ensure that x is a valid dataset.

    Parameters
    ==========

    x : ndarray
        The array to check
    allow_1d : bool, optional
        Allow the dataset to be a single time series
    allow_2d : bool, optional
        Allow the dataset to be a univariate dataset
    allow_3d : bool, optional
        Allow the dataset to be a multivariate dataset

    Returns
    =======
    ndarray
        The checked array
    """
    if not allow_1d and x.ndim == 1:
        raise ValueError("1d-array is not allowed")
    if not allow_2d and x.ndim == 2:
        raise ValueError("2d-array is not allowed")
    if not allow_3d and x.ndim == 3:
        raise ValueError("3d-array is not allowed")

    if x.ndim > 3:
        raise ValueError("invalid array dim (%d)" % x.ndim)

    if x.ndim == 1:
        x = x.reshape(1, -1)

    if x.ndim == 3 and x.shape[1] == 1:
        x = x.reshape(x.shape[0], x.shape[x.ndim - 1])
    last_stride = x.strides[x.ndim - 1] // x.itemsize
    if (x.ndim > 1 and last_stride != 1) or not x.flags.carray:
        x = np.ascontiguousarray(x)

    return x.astype(np.double, copy=False)


cdef class Dataset:

    def __cinit__(self, arr):
        """Construct a new time series dataset from a ndarray. The ndarray must remain
        in scope for the lifetime of the dataset. The dataset is invalid once the
        ndarray has been garbage collected.
        
        Parameters
        ==========

        data : ndarray of shape (n_samples, n_timestep) or (n_samples, n_dim, n_timestep)
            The data array to use as dataset.

            The last dimension of data must be in c-order.

            Use `wildboar.utils.data.check_dataset` to ensure a valid array.
        """
        self.data = memoryview(arr, PyBUF_STRIDES & PyBUF_C_CONTIGUOUS)

        if self.data.view.ndim < 2 or self.data.view.ndim > 3:
            raise ValueError("ndim {0} < 2 or {0} > 3".format(self.data.view.ndim))
        
        self.n_samples = <Py_ssize_t> self.data.view.shape[0]
        self.n_timestep = <Py_ssize_t> self.data.view.shape[self.data.view.ndim - 1]
        self.sample_stride = <Py_ssize_t> (
            self.data.view.strides[0] / <Py_ssize_t> self.data.view.itemsize
        )
        
        cdef Py_ssize_t timestep_stride
        if self.n_timestep == 1:
            timestep_stride = 1
        else:
            timestep_stride = <Py_ssize_t> (
                self.data.view.strides[self.data.view.ndim - 1] / <Py_ssize_t> self.data.view.itemsize
            )
        
        if timestep_stride != 1:
            raise ValueError(
                "timestep_stride is invalid (%d != 1)" % timestep_stride,
            )

        if self.data.view.ndim == 3:
            self.n_dims = <Py_ssize_t> self.data.view.shape[self.data.view.ndim - 2]
            if self.n_dims == 1:
                self.dim_stride = 1
            else:
                self.dim_stride = <Py_ssize_t> (
                    self.data.view.strides[self.data.view.ndim - 2] / <Py_ssize_t> self.data.view.itemsize
                )
        else:
            self.n_dims = 1
            self.dim_stride = 0

    cdef double* get_sample(self, Py_ssize_t i, Py_ssize_t dim) nogil:
        cdef Py_ssize_t offset = self.sample_stride * i + self.dim_stride * dim
        return <double*> self.data.view.buf + offset