import math
import numbers

import numpy as np
from sklearn.utils.validation import check_scalar

from ..utils.validation import _check_ts_array, check_array, check_type
from . import _cmatrix_profile
from ._distance import _format_return


def matrix_profile(  # noqa: PLR0912
    x,
    y=None,
    *,
    window=5,
    dim=0,
    exclude=None,
    n_jobs=-1,
    return_index=False,
):
    """
    Compute the matrix profile.

    - If only ``x`` is given, compute the similarity self-join of every subsequence in
      ``x`` of size ``window`` to its nearest neighbor in `x` excluding trivial matches
      according to the ``exclude`` parameter.
    - If both ``x`` and ``y`` are given, compute the similarity join of every
      subsequenec in ``y`` of size ``window`` to its nearest neighbor in ``x`` excluding
      matches according to the ``exclude`` parameter.

    Parameters
    ----------
    x : array-like of shape (n_timestep, ), (n_samples, xn_timestep) or \
        (n_samples, n_dim, xn_timestep)
        The first time series.
    y : array-like of shape (n_timestep, ), (n_samples, yn_timestep) or \
        (n_samples, n_dim, yn_timestep), optional
        The optional second time series. y is broadcast to the shape of x if possible.
    window : int or float, optional
        The subsequence size, by default 5

        - if float, a fraction of `y.shape[-1]`.
        - if int, the exact subsequence size.
    dim : int, optional
        The dim to compute the matrix profile for, by default 0.
    exclude : int or float, optional
        The size of the exclusion zone. The default exclusion zone is  0.2 for
        similarity self-join and 0.0 for similarity join.

        - if float, expressed as a fraction of the windows size.
        - if int, exact size (0 >= exclude < window).
    n_jobs : int, optional
        The number of jobs to use when computing the profile.
    return_index : bool, optional
        Return the matrix profile index.

    Returns
    -------
    mp : ndarray of shape (profile_size, ) or (n_samples, profile_size)
        The matrix profile.
    mpi : ndarray of shape (profile_size, ) or (n_samples, profile_size), optional
        The matrix profile index.

    Notes
    -----
    The `profile_size` depends on the input.

    - If `y` is `None`, `profile_size` is  ``x.shape[-1] - window + 1``
    - If `y` is not `None`, `profile_size` is ``y.shape[-1] - window + 1``

    References
    ----------
    Yeh, C. C. M. et al. (2016).
        Matrix profile I: All pairs similarity joins for time series: a unifying view
        that includes motifs, discords and shapelets. In 2016 IEEE 16th international
        conference on data mining (ICDM)
    """
    x = check_array(x, allow_3d=True, ensure_2d=False)

    if y is not None:
        y = check_array(y, allow_3d=True, ensure_2d=False)
        if x.ndim > 1:
            y = np.broadcast_to(y, x.shape)

        if x.ndim != y.ndim:
            raise ValueError("Both x and y must have the same dimensionality")
        if x.ndim != 1 and x.shape[0] != y.shape[0]:
            raise ValueError("Both x and y must have the same number of samples")
        if x.ndim > 2 and x.shape[1] != y.shape[1]:
            raise ValueError("Both x and y must have the same number of dimensions")
        if not y.shape[-1] <= x.shape[-1]:
            raise ValueError(
                "y.shape[-1] > x.shape[-1]. If you want to compute the matrix profile "
                "of the similarity join of YX, swap the order of inputs."
            )
        exclude = exclude if exclude is not None else 0.0
    else:
        y = x
        exclude = exclude if exclude is not None else 0.2

    if x.ndim > 2 and not 0 <= dim < x.shape[1]:
        raise ValueError("Invalid dim (%d)" % x.shape[1])

    check_type(window, "window", (numbers.Integral, numbers.Real))
    check_type(exclude, "exclude", (numbers.Integral, numbers.Real))
    if isinstance(window, numbers.Integral):
        check_scalar(
            window,
            "window",
            numbers.Integral,
            min_val=1,
            max_val=min(y.shape[-1], x.shape[-1]),
        )
    elif isinstance(window, numbers.Real):
        check_scalar(
            window,
            "window",
            numbers.Real,
            min_val=0,
            max_val=1,
            include_boundaries="right",
        )
        window = math.ceil(window * y.shape[-1])

    if isinstance(exclude, numbers.Integral):
        check_scalar(exclude, "exclude", numbers.Integral, min_val=1)
    elif isinstance(exclude, numbers.Real):
        check_scalar(
            exclude,
            "exclude",
            numbers.Real,
            min_val=0,
        )
        exclude = math.ceil(window * exclude)

    mp, mpi = _cmatrix_profile._paired_matrix_profile(
        _check_ts_array(x),
        _check_ts_array(y),
        window,
        dim,
        exclude,
        n_jobs,
    )

    if return_index:
        return _format_return(mp, 2, x.ndim), _format_return(mpi, 2, x.ndim)
    else:
        return _format_return(mp, 2, x.ndim)
