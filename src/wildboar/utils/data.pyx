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

# Authors: Isak Samsten

cimport numpy as np

import warnings

import numpy as np


cdef class Dataset:

    def __cinit__(self, np.ndarray data):
        """Construct a new time series dataset from a ndarray
        
        Parameters
        ==========

        data : ndarray of shape (n_samples, n_timestep) or (n_samples, n_dim, n_timestep)
            The data array to use as dataset
        """
        if data.ndim < 2 or data.ndim > 3:
            raise ValueError("ndim {0} < 2 or {0} > 3".format(data.ndim))
        
        data = np.ascontiguousarray(data, dtype=np.float64)
        self.n_samples = <Py_ssize_t> data.shape[0]
        self.n_timestep = <Py_ssize_t> data.shape[data.ndim - 1]
        self.data = <double*> data.data
        self.sample_stride = <Py_ssize_t> (data.strides[0] / <Py_ssize_t> data.itemsize)
        cdef Py_ssize_t timestep_stride = <Py_ssize_t> (data.strides[data.ndim - 1] / <Py_ssize_t> data.itemsize)
        if timestep_stride != 1:
            warnings.warn(
                "timestep_stride (%d) detected. Please report bug." % timestep_stride, 
                UserWarning,
            )

        if data.ndim == 3:
            self.n_dims = <Py_ssize_t> data.shape[data.ndim - 2]
            self.dim_stride = <Py_ssize_t> (data.strides[data.ndim - 2] / <Py_ssize_t> data.itemsize)
        else:
            self.n_dims = 1
            self.dim_stride = 0


    cdef double* get_sample(self, Py_ssize_t i, Py_ssize_t dim=0) nogil:
        cdef Py_ssize_t offset = self.sample_stride * i + self.dim_stride * dim
        return self.data + offset