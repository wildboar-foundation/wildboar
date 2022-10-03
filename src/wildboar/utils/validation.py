# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np
from sklearn.utils.validation import _check_estimator_name, _check_y
from sklearn.utils.validation import check_array as sklearn_check_array
from sklearn.utils.validation import check_consistent_length

import wildboar as wb


def _num_timesteps_dim(dim):
    type_ = type(dim)
    if type_.__module__ == "builtins":
        type_name = type_.__qualname__
    else:
        type_name = f"{type_.__module__}.{type_.__qualname__}"
    message = (
        f"Unable to find the number of timesteps from dimension of type {type_name}"
    )
    if not hasattr(dim, "__len__") and not hasattr(dim, "shape"):
        if not hasattr(dim, "__array__"):
            raise TypeError(message)
        dim = np.asarray(dim)

    if hasattr(dim, "shape"):
        if not hasattr(dim, "__len__") or len(dim.shape) != 1:
            message += f" with shape {dim.shape}"
            raise TypeError(message)

        return dim.shape[0]

    if len(dim) > 0:
        if (
            hasattr(dim, "__len__")
            or hasattr(dim, "shape")
            or hasattr(dim, "__array__")
        ):
            raise TypeError(message)

    return len(dim)


def _num_timesteps(X):
    """Return the number of timesteps and dimensions of ``X``

    Parameters
    ----------
    X : object
        The object to guess the number of timesteps and dimensions

    Returns
    -------
    n_timesteps : int
        The number of timesteps

    n_dims : int
        The number of dimensions
    """
    type_ = type(X)
    if type_.__module__ == "builtins":
        type_name = type_.__qualname__
    else:
        type_name = f"{type_.__module__}.{type_.__qualname__}"
    message = f"Unable to find the number of timesteps from X of type {type_name}"
    if not hasattr(X, "__len__") and not hasattr(X, "shape"):
        if not hasattr(X, "__array__"):
            raise TypeError(message)
        # Only convert X to a numpy array if there is no cheaper, heuristic
        # option.
        X = np.asarray(X)

    if hasattr(X, "shape"):
        if not hasattr(X.shape, "__len__") or len(X.shape) <= 1 or len(X.shape) > 3:
            message += f" with shape {X.shape}"
            raise TypeError(message)
        return X.shape[-1], X.shape[1] if len(X.shape) == 3 else 1

    first_sample = X[0]

    # Do not consider an array-like of strings or dicts to be a 2D array
    if isinstance(first_sample, (str, bytes, dict)):
        message += f" where the samples are of type {type(first_sample).__qualname__}"
        raise TypeError(message)

    try:
        possibly_first_dim = first_sample[0]
        if hasattr(possibly_first_dim, "__len__") or hasattr(
            possibly_first_dim, "shape"
        ):
            return _num_timesteps_dim(possibly_first_dim), len(first_sample)

        # If X is a list of lists, for instance, we assume that all nested
        # lists have the same length without checking or converting to
        # a numpy array to keep this function call as cheap as possible.
        return len(first_sample), 1
    except Exception as err:
        raise TypeError(message) from err


def check_X_y(
    x,
    y,
    *,
    allow_3d=False,
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
        allow_3d=allow_3d,
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
    allow_3d=False,
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
    allow_3d : bool, optional
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

    # Never force_all_finite. We always force all finite.
    if "force_all_finite" in kwargs:
        del kwargs["force_all_finite"]

    if allow_3d:
        if "ensure_2d" in kwargs and kwargs.pop("ensure_2d"):
            raise ValueError("ensure_2d=True and allow_3d=True are incompatible")

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
