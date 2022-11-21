# cython: language_level=3
# cython: boundscheck=False

# Authors: Isak Samsten
# License: BSD 3 clause
import numpy as np

__all__ = [
    "Dataset"
]

cdef TSArray check_dataset(x):
    """Ensure that x is a valid dataset.

    Parameters
    ==========

    x : ndarray
        The array to check

    Returns
    =======
    ndarray
        The checked array
    """
    if x.ndim > 3 or x.ndim < 1:
        raise ValueError("invalid array dim (%d)" % x.ndim)

    if x.ndim == 1:
        x = np.expand_dims(x, axis=(0, 1))
    elif x.ndim == 2:
        x = np.expand_dims(x, axis=1)

    last_stride = x.strides[2] // x.itemsize
    if last_stride != 1 or not x.flags.carray:
        x = np.ascontiguousarray(x)

    return x.astype(float, copy=False)


cdef class Dataset:

    def __cinit__(self, object array):
        self.view = check_dataset(array)
        self.n_samples = self.view.shape[0]
        self.n_timestep = self.view.shape[2]
        self.n_dims = self.view.shape[1]

    cdef const double* get_sample(self, Py_ssize_t i, Py_ssize_t dim) nogil:
        return &self.view[i, dim, 0]

    @property
    def array(self):
        return np.asarray(self.view)