import numpy as np
from sklearn.utils.validation import _check_estimator_name, _check_y
from sklearn.utils.validation import check_array as sklearn_check_array
from sklearn.utils.validation import check_consistent_length

import wildboar as wb


def check_X_y(
    x,
    y,
    *,
    allow_multivariate=False,
    ensure_min_samples=1,
    allow_eos=False,
    allow_nan=False,
    contiguous=True,
    dtype=float,
    y_numeric=False,
    y_contiguous=True,
    multi_output=False,
    estimator=None,
):
    if y is None:
        if estimator is None:
            estimator_name = "estimator"
        else:
            estimator_name = _check_estimator_name(estimator)
        raise ValueError(
            f"{estimator_name} requires y to be passed, but the target y is None"
        )
    x = check_array(
        x,
        allow_multivariate=allow_multivariate,
        ensure_min_samples=ensure_min_samples,
        allow_eos=allow_eos,
        allow_nan=allow_nan,
        contiguous=contiguous,
        ensure_1d=False,
        dtype=dtype,
    )

    if y_numeric:
        y_dtype = np.float64
    else:
        y_dtype = None

    if y_contiguous:
        y = np.ascontiguousarray(y, dtype=y_dtype)

    y = _check_y(y, multi_output=multi_output, y_numeric=y_numeric, estimator=estimator)
    check_consistent_length(x, y)
    return x, y


def check_array(
    x,
    allow_multivariate=False,
    ensure_1d=False,
    allow_eos=False,
    allow_nan=False,
    contiguous=True,
    **kwargs,
):
    """Wrapper to check array

    Parameters
    ----------
    x : ndarray
        The array to check
    allow_multivariate : bool, optional
        If 3d arrays are allowed, by default False
    ensure_1d: bool, optional
        Ensure that the array has only one dimension.
    allow_eos : bool, optional
        If unequal length series are allowed
    allow_nan : bool, optional
        If NaN values are allowed
    contiguous : bool, optional
        Ensure that the array is in c-order.
    kwargs : dict
        Additional arguments passed to `sklearn.utils.check_array`

    Returns
    -------
    ndarray
        The checked array
    """
    if contiguous:
        order = kwargs.get("order", None)
        if order is not None and order.lower() != "c":
            raise ValueError("order=%r and contiguous=True are incompatible")
        kwargs["order"] = "C"

    if allow_multivariate:
        if "ensure_2d" in kwargs and kwargs.pop("ensure_2d"):
            raise ValueError(
                "ensure_2d=True and allow_multivariate=True are incompatible"
            )

        if "allow_nd" in kwargs and not kwargs.pop("allow_nd"):
            raise ValueError(
                "allow_nd=False and allow_multivaraite=True are incompatible"
            )
        x = sklearn_check_array(
            x,
            ensure_2d=False,
            allow_nd=True,
            force_all_finite=False,
            **kwargs,
        )
        if x.ndim == 0:
            raise ValueError(
                "Expected 2D or 3D array, got scalar array instead:\narray={}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single timestep or array.reshape(1, -1) "
                "if it contains a single sample.".format(x)
            )
        if x.ndim == 1:
            raise ValueError(
                "Expected 2D or 3D array, got 1D array instead:\narray={}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single timestep or array.reshape(1, -1) "
                "if it contains a single sample.".format(x)
            )
        if x.ndim > 3:
            raise ValueError(
                "Expected 2D or 3D array, got {}D array instead:\narray={}.\n".format(
                    x.ndim, x
                )
            )
    elif ensure_1d:
        if "ensure_2d" in kwargs and kwargs.pop("ensure_2d"):
            raise ValueError("ensure_2d=True and ensure_1d=True are incompatible")
        x = np.squeeze(
            sklearn_check_array(
                x, ensure_2d=False, allow_nd=False, force_all_finite=False, **kwargs
            )
        )
        if x.ndim == 0:
            raise ValueError(
                "Expected 2D or 3D array, got scalar array instead:\narray={}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single timestep or array.reshape(1, -1) "
                "if it contains a single sample.".format(x)
            )

        if x.ndim > 1:
            raise ValueError(
                "Expected 1D or 2D array with an empty dimension, "
                "got {}D array instead:\narray={}.\n".format(x.ndim, x)
            )
    else:
        x = sklearn_check_array(x, force_all_finite=False, **kwargs)

    if np.issubdtype(x.dtype, np.double):
        if not allow_eos and wb.iseos(x).any():
            raise ValueError("Expected time series of equal length.")

        if not allow_nan and np.isnan(x).any():
            raise ValueError("Input contains NaN.")

        if np.isposinf(x).any():
            raise ValueError("Input contains infinity.")

    return x
