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


cdef struct TSDatabase:
    Py_ssize_t n_samples  # the number of samples
    Py_ssize_t n_timestep  # the number of timesteps
    Py_ssize_t n_dims

    double *data  # the data
    Py_ssize_t sample_stride  # the stride for samples
    Py_ssize_t timestep_stride  # the `feature` stride
    Py_ssize_t dim_stride  # the dimension stride


cdef TSDatabase ts_database_new(np.ndarray X)