import math
import numbers
import warnings

import numpy as np
from sklearn.utils._param_validation import Interval, StrOptions, validate_params

from ..utils.validation import _check_ts_array, check_array
from . import _cmatrix_profile
from ._distance import _format_return


def paired_matrix_profile(  # noqa: PLR0912
    X,
    Y=None,
    *,
    window=5,
    dim=0,
    exclude=None,
    n_jobs=-1,
    return_index=False,
):
    """
    Compute the matrix profile.

    - If only `X` is given, compute the similarity self-join of every subsequence in
      `X` of size ``window`` to its nearest neighbor in `X` excluding trivial matches
      according to the `exclude` parameter.
    - If both `X` and `Y` are given, compute the similarity join of every
      subsequenec in `X` of size `window` to its nearest neighbor in `Y` excluding
      matches according to the `exclude` parameter.

    Parameters
    ----------
    X : array-like of shape (n_timestep, ), (n_samples, x_timestep) or \
        (n_samples, n_dim, x_timestep)
        The first time series.
    Y : array-like of shape (n_timestep, ), (n_samples, y_timestep) or \
        (n_samples, n_dim, y_timestep), optional
        The optional second time series. Y is broadcast to the shape of X if possible.
    window : int or float, optional
        The subsequence size, by default 5

        - if float, a fraction of y_timestep
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
    The `profile_size` is  ``X.shape[-1] - window + 1``.

    References
    ----------
    Yeh, C. C. M. et al. (2016).
        Matrix profile I: All pairs similarity joins for time series: a unifying view
        that includes motifs, discords and shapelets. In 2016 IEEE 16th international
        conference on data mining (ICDM)
    """
    X = check_array(X, allow_3d=True, ensure_2d=False)

    if Y is not None:
        Y = check_array(Y, allow_3d=True, ensure_2d=False)
        if X.ndim > 1:
            Y = np.broadcast_to(Y, X.shape)

        if X.ndim != Y.ndim:
            raise ValueError("Both x and y must have the same dimensionality")
        if X.ndim != 1 and X.shape[0] != Y.shape[0]:
            raise ValueError("Both x and y must have the same number of samples")
        if X.ndim > 2 and X.shape[1] != Y.shape[1]:
            raise ValueError("Both x and y must have the same number of dimensions")
        if not Y.shape[-1] <= X.shape[-1]:
            raise ValueError(
                "y.shape[-1] > x.shape[-1]. If you want to compute the matrix profile "
                "of the similarity join of YX, swap the order of inputs."
            )
        exclude = exclude if exclude is not None else 0.0
    else:
        Y = X
        exclude = exclude if exclude is not None else 0.2

    if not isinstance(window, numbers.Integral):
        window = math.ceil(window * Y.shape[-1])

    if not (0 < window <= min(X.shape[-1], Y.shape[-1])):
        raise ValueError("window must be 0 < window <= min(x_timestep, y_timestep)")

    if not isinstance(exclude, numbers.Integral):
        exclude = math.ceil(window * exclude)

    if X.ndim > 2 and X.ndim != Y.ndim and X.shape[1] != Y.shape[1]:
        raise ValueError("Both X and Y must have the same number of dimensions")

    if X.ndim > 2 and not 0 <= dim < X.shape[1]:
        raise ValueError("dim must be less than n_dims")

    mp, mpi = _cmatrix_profile._paired_matrix_profile(
        _check_ts_array(Y),
        _check_ts_array(X),
        window,
        dim,
        exclude,
        n_jobs,
    )

    if return_index:
        return _format_return(mp, 2, X.ndim), _format_return(mpi, 2, X.ndim)
    else:
        return _format_return(mp, 2, X.ndim)


@validate_params(
    {
        "x": ["array-like"],
        "y": [None, "array-like"],
        "dim": [Interval(numbers.Integral, 0, None, closed="left")],
        "window": [
            Interval(numbers.Integral, 1, None, closed="left"),
            Interval(numbers.Real, 0, 1, closed="both"),
        ],
        "exclude": [
            None,
            Interval(numbers.Integral, 1, None, closed="left"),
            Interval(numbers.Real, 0, 1, closed="both"),
        ],
        "kind": [StrOptions({"default", "paired", "warn"}, deprecated={"warn"})],
        "return_index": [bool],
        "n_jobs": [None, numbers.Integral],
    },
    prefer_skip_nested_validation=True,
)
def matrix_profile(  # noqa: PLR0912
    X,
    Y=None,
    *,
    dim=0,
    window=5,
    exclude=None,
    kind="warn",
    return_index=False,
    n_jobs=None,
):
    """
    Compute the matrix profile of every subsequence in X.

    If Y is given compute the metrix profile of every subsequence in X finding
    the minimum distance in any time series in Y; othervise finding the minimum
    distance in any time series in X. The former corresponds to a self-join and
    the latter to an AB join.

    The output approximately corresponds to that of
    :func:`~wildboar.distance.matrix_profile` where `X.flatten()` but without
    computing the distance where two time series overlap. The outputs exactly
    correspond when ``X.shape[0] == 1``.

    Parameters
    ----------
    X : array-like of shape (x_samples, x_timestep)
        The time series for which the matrix profile is computed.
    Y : array-like of shape (y_samples, y_timestep), optional
        The time series used to annotate `X`. If None, `X` is used to annotate.
    dim : int, optional
        The dimension.
    window : int or float, optional
        The window size.

        - If float, the window size is a fraction of `x_timestep`.
        - If int, the window size is exact.
    exclude : int or float, optional
        The exclusion zone.

        - If float, the exclusion zone is a fraction of `window`.
        - If int, the exclusion zone is exact.
        - If None, the exclusion zone is determined automatically. If Y is None,
          (self-join) the value is `0.2`, otherwise (AB-join) the value is
          `0.0`.
    kind : {"paired", "default"}, optional
        The kind of matrix profile.

        - if "paired", compute the matrix profile for each time series in X
          optionally annotated with each time series in Y.
        - if "default", compute the matrix profile for every subsequence in
          every time series in X optional annotated with every time series in
          Y.
    return_index : bool, optional
        Return the matrix profile index.
    n_jobs : int, optional
        The number of parallel jobs.

    Returns
    -------
    mp : ndarray of shape (x_samples, profile_size)
        The matrix profile.
    (mpi_sample, mpi_start) : ndarray of shape (x_samples, profile_size), optional
        The matrix profile index sample and start positions. Returned if
        `return_index=True`.
    """
    if kind in ("warn", "paired"):
        if kind == "warn":
            # TODO(1.4)
            warnings.warn(
                "matrix_profile has changed name to paired_matrix_profile in 1.3 "
                "and the default behavior will change in 1.4",
                FutureWarning,
            )
            if Y is not None:
                X, Y = Y, X

        return paired_matrix_profile(
            X, Y, window=window, dim=dim, exclude=exclude, n_jobs=n_jobs
        )
    else:
        if Y is None:
            Y = X

        if X is Y:
            X = check_array(X, ensure_ts_array=True)
            Y = X
            if exclude is None:
                exclude = 0.2
        else:
            X = check_array(X, ensure_ts_array=True)
            Y = check_array(Y, ensure_ts_array=True)
            if exclude is None:
                exclude = 0.0

        if not isinstance(window, numbers.Integral):
            window = math.ceil(window * Y.shape[-1])

        if not (0 < window <= min(X.shape[-1], Y.shape[-1])):
            raise ValueError("window must be 0 < window <= min(x_timestep, y_timestep)")

        if not isinstance(exclude, numbers.Integral):
            exclude = math.ceil(window * exclude)

        if X.shape[1] != Y.shape[1]:
            raise ValueError("Both X and Y must have the same number of dimensions")

        if not 0 <= dim < X.shape[1]:
            raise ValueError("dim must be less than n_dims")

        mp, mpi = _cmatrix_profile._flat_matrix_profile_join(
            X, Y, window, dim, exclude, n_jobs
        )
        if return_index:
            return mp, (mpi[:, :, 0], mpi[:, :, 1])
        else:
            return mp
