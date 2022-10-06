# Authors: Isak Samsten
# License: BSD 3 clause
import math
import warnings

import numpy as np

from ..distance import matrix_profile
from ..utils.decorators import array_or_scalar
from ..utils.validation import check_array


@array_or_scalar()
def segment(
    x=None,
    *,
    mpi=None,
    n_segments=1,
    window=0.1,
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

    n_segments : int, optional
        The number of segmentations to identify

    window : int or float, optional
        The window size. The window parameter is ignored if mpi is not None.

        - if float, a fraction of n_timestep
        - if int, the exact window size

    exclude : float, optional
        The self-join exclusion for the matrix profile. Ignored if `mpi` is given.

    boundry : float, optional
        The region around an identified segmentation that is ignored when searching
        for subsequent segmentations

    return_arc_curve : bool, optional
        Return the arc curve.

    Returns
    -------
    segments : ndarray of shape (n_samples, n_segments), (n_segments) or int
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
        raise ValueError("Either x or mpi must be given")
    elif x is not None:
        if mpi is not None:
            raise ValueError("Both x and mpi cannot be given")

        x = check_array(np.atleast_2d(x), allow_3d=False)
        _, mpi = matrix_profile(x, window=window, exclude=exclude, return_index=True)
        mpi = np.atleast_2d(mpi)
    else:
        mpi = check_array(np.atleast_2d(mpi))

    boundry = math.ceil(window * boundry)
    segments = -np.ones((mpi.shape[0], n_segments), dtype=np.intp)
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

        for j in range(n_segments):
            segments[i, j] = np.argmin(arc_curve)
            if arc_curve[segments[i, j]] == 1.0:
                warnings.warn(
                    f"No more segments for sample={i} (segment={j}). All remaining"
                    f"segments are invalid and set to -1.",
                    UserWarning,
                )
                break
            start = max(segments[i, j] - boundry, 0)
            end = min(segments[i, j] + boundry, arc_curve.shape[0])
            arc_curve[start:end] = 1.0

    if return_arc_curve:
        return segments, arc_curves
    else:
        return segments
