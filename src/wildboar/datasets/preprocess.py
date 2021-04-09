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
import numpy as np


def named_preprocess(name):
    if name == "standardize":
        return standardize
    else:
        raise ValueError("%s does not exists" % name)


def standardize(x):
    """Standardize x (along the time dimension) to have zero mean and unit standard deviation

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_timestep)
        The dataset

    Returns
    -------
    x : ndarray of shape (n_samples, n_timestep)
        The standardized dataset
    """
    return (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
