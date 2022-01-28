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
import math
import warnings

import numpy as np

from wildboar.distance import matrix_profile
from wildboar.utils import check_array
from wildboar.utils.decorators import array_or_scalar


@array_or_scalar()
def regimes(
    x=None,
    mpi=None,
    *,
    n_regimes=1,
    window=5,
    exclude=0.2,
    boundry=1.0,
    return_arc_curve=False,
):
    """Find change regimes in a time series.

    Parameters
    ----------
    x : array-like of shape (n_samples, n_timestep) or (n_timestep, ), optional
        The time series. If x is given, the matrix profile of x is computed.

    mpi : array-like of shape (n_samples, profile_size) or (profile_size), optional
        The matrix profile index. Must be given unless x is given.

    n_regimes : int, optional
        The number of segmentations to identify

    window : int, optional
        The window size. Ignored if `mpi` is given.

    exclude : float, optional
        The self-join exclusion for the matrix profile. Ignored if `mpi` is given.

    boundry : float, optional
        The region around an identified segmentation that is ignored when searching
        for subsequent segmentations

    return_arc_curve : bool, optional
        Return the arc curve.

    Returns
    -------
    segments : ndarray of shape (n_samples, n_regimes), (n_regimes) or int
        The start index of a segment

    arc_curves : ndarray of shape (n_samples, profile_size) or (profile_size, )
        The arc curves

    See also
    --------
    wildboar.distance.matrix_profile : compute the matrix profile

    References
    ----------
    Gharghabi, Shaghayegh, et al. (2017)
        Matrix profile VIII: domain agnostic online semantic segmentation at superhuman
        performance levels. In proceedings of International Conference on Data Mining
    """
    if x is None and mpi is None:
        raise ValueError("either x or mpi must be given")
    if x is not None:
        if mpi is not None:
            raise ValueError("both x and mpi cannot be given")
        x = check_array(np.atleast_2d(x), allow_multivariate=False)
        _, mpi = matrix_profile(x, window=window, exclude=exclude, return_index=True)
        mpi = np.atleast_2d(mpi)
    else:
        mpi = check_array(np.atleast_2d(mpi))

    boundry = math.ceil(window * boundry)
    regimes = np.empty((mpi.shape[0], n_regimes), dtype=np.intp)
    if return_arc_curve:
        arc_curves = np.empty(mpi.shape, dtype=np.double)

    index = np.arange(mpi.shape[-1])
    arc_curve_normalize = 2 * index * (mpi.shape[-1] - index) / mpi.shape[-1]
    arc_curve_normalize[0] = 10e-10
    for i in range(mpi.shape[0]):
        arc_curve = np.minimum(
            np.cumsum(
                np.bincount(np.minimum(index, mpi[i]), minlength=mpi.shape[-1])
                - np.bincount(np.maximum(index, mpi[i]), minlength=mpi.shape[-1])
            )
            / arc_curve_normalize,
            1,
        )
        arc_curve[:boundry] = 1.0
        arc_curve[-boundry:] = 1.0

        if return_arc_curve:
            arc_curves[i, :] = arc_curve

        for j in range(n_regimes):
            regimes[i, j] = np.argmin(arc_curve)
            if arc_curve[regimes[i, j]] == 1.0:
                warnings.warn(
                    f"no more regimes for sample={i} (regime={j}) all remaining"
                    "regimes are invalid",
                    UserWarning,
                )
                break
            start = max(regimes[i, j] - boundry, 0)
            end = min(regimes[i, j] + boundry, arc_curve.shape[0])
            arc_curve[start:end] = 1.0

    if return_arc_curve:
        return regimes, arc_curves
    else:
        return regimes
