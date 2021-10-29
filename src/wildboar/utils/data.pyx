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


cdef Dataset dataset_new(np.ndarray data):
    """Construct a new time series dataset from a ndarray
    
    Parameters
    ==========

    data : ndarray of shape (n_samples, n_timestep) or (n_samples, n_dim, n_timestep)
        The data array to use as dataset

    Returns
    =======

    dataset : Dataset 
        A dataset with c-contigous data
    """
    if data.ndim < 2 or data.ndim > 3:
        raise ValueError("ndim {0} < 2 or {0} > 3".format(data.ndim))
    
    data = np.ascontiguousarray(data, dtype=np.float64)

    cdef Dataset sd
    sd.n_samples = <Py_ssize_t> data.shape[0]
    sd.n_timestep = <Py_ssize_t> data.shape[data.ndim - 1]
    sd.data = <double*> data.data
    sd.sample_stride = <Py_ssize_t> (data.strides[0] / <Py_ssize_t> data.itemsize)
    cdef Py_ssize_t timestep_stride = <Py_ssize_t> (data.strides[data.ndim - 1] / <Py_ssize_t> data.itemsize)
    if timestep_stride != 1:
        warnings.warn(
            "timestep_stride (%d) detected. Please report bug." % timestep_stride, 
            UserWarning,
        )

    if data.ndim == 3:
        sd.n_dims = <Py_ssize_t> data.shape[data.ndim - 2]
        sd.dim_stride = <Py_ssize_t> (data.strides[data.ndim - 2] / <Py_ssize_t> data.itemsize)
    else:
        sd.n_dims = 1
        sd.dim_stride = 0

    return sd