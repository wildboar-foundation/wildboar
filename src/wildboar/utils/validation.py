# Authors: Isak Samsten
# License: BSD 3 clause

import numbers

import numpy as np
from sklearn.utils.validation import _check_estimator_name, _check_y
from sklearn.utils.validation import check_array as sklearn_check_array
from sklearn.utils.validation import check_consistent_length

import wildboar as wb


def check_type(x, name, target_type, required=True):
    """Check that the type of x is of a target type.

    Parameters
    ----------
    x : object
        The object to check.

    name : str
        The name of the parameter.

    target_type : type or tuple
        The required type(s) of x.

    required : bool, optional
        If required=False, None is an allowed.
    """

    def type_name(t):
        module = t.__module__
        qualname = t.__qualname__
        if module == "builtins":
            return qualname
        elif t == numbers.Real:
            return "float"
        elif t == numbers.Integral:
            return "int"
        return f"{module}.{qualname}"

    if isinstance(target_type, tuple):
        types_str = ", ".join(type_name(t) for t in target_type)
        target_type_str = f"{{{types_str}}}"
    else:
        target_type_str = type_name(target_type)

    if x is None and required:
        raise TypeError(f"{name} must be an instance of {target_type_str}, not None")
    if x is None and not required:
        return

    if not isinstance(x, target_type):
        raise TypeError(
            f"{name} must be an instance of {target_type_str}, not"
            f" {type(x).__qualname__}."
        )


def check_option(dict, key, name):
    if key in dict:
        return dict[key]
    else:
        keys = ["'%s'" % key for key in sorted(list(dict.keys()))]
        if len(keys) == 1:
            msg = f"{name} must be {keys[0]}, got {key}"
        else:
            msg = f"{name} must be {', '.join(keys[:-1])} or {keys[-1]}, got {key}"

        raise ValueError(msg)


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
    dtype=float,
    order="C",
    copy=False,
    ensure_2d=True,
    allow_3d=False,
    allow_nd=False,
    force_all_finite=True,
    multi_output=False,
    ensure_min_samples=1,
    ensure_min_timesteps=1,
    ensure_min_dims=1,
    allow_eos=False,
    y_numeric=False,
    y_contiguous=True,
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
        allow_nd=allow_nd,
        ensure_min_samples=ensure_min_samples,
        ensure_min_dims=ensure_min_dims,
        ensure_min_timesteps=ensure_min_timesteps,
        force_all_finite=force_all_finite,
        allow_eos=allow_eos,
        order=order,
        copy=copy,
        ensure_2d=ensure_2d,
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
    array,
    *,
    dtype="numeric",
    order="C",
    copy=False,
    ravel_1d=False,
    ensure_2d=True,
    allow_3d=False,
    allow_nd=False,
    force_all_finite=True,
    ensure_min_samples=1,
    ensure_min_timesteps=1,
    ensure_min_dims=1,
    estimator=None,
    input_name="",
    allow_eos=False,
):
    """Delegate array validation to scikit-learn
    :func:`sklearn.utils.validation.check_array` with wildboar defaults and conventions.

    - we optionally allow end-of-sequence identifiers
    - by default we convert arrays to c-order
    - we optionally specifically allow for 3d-arrays
    - we never allow for sparse arrays

    By default, the input is checked to be a non-empty 2D array in c-order containing
    only finite values, with at least 1 sample, 1 timestep and 1 dimension. If the dtype
    of the array is object, attempt converting to float, raising on failure.

    Parameters
    ----------
    array : object
        Input object to check / convert.

    dtype : 'numeric', type, list of type or None, optional
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : {'F', 'C'} or None, optional
        Whether an array will be forced to be fortran or c-style.
        When order is None, then if copy=False, nothing is ensured
        about the memory layout of the output array; otherwise (copy=True)
        the memory layout of the returned array is kept as close as possible
        to the original array.

    copy : bool, optional
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    ravel_1d : bool, optional
        Whether to ravel 1d arrays or column vectors, it the array is neither an
        error is raised.

    ensure_2d : bool, optional
        Whether to raise a value error if array is not 2D.

    allow_3d : bool, optional
        Wheter to allow array.ndim == 3

    allow_nd : bool, optional
        Whether to allow array.ndim > 2.

    force_all_finite : bool or 'allow-nan', default=True
        Whether to raise an error on np.inf, np.nan, pd.NA in array. The
        possibilities are:

        - True: Force all values of array to be finite.
        - False: accepts np.inf, np.nan, pd.NA in array.
        - 'allow-nan': accepts only np.nan and pd.NA values in array. Values
          cannot be infinite.

        If allow_eos=True, -np.inf is allowed despite force_all_finite=True.

    ensure_min_samples : int, optional
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.

    ensure_min_timesteps : int, optional
        Make sure that the 2D array has some minimum number of timesteps
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.

    ensure_min_dims : int, optional
        Make sure that the array has a minimum number of dimensions. Setting to 0
        disables this check.

    estimator : str or estimator instance, default=None
        If passed, include the name of the estimator in warning messages.

    input_name : str, default=""
        The data name used to construct the error message.

    Returns
    -------
    array_converted : object
        The converted and validated array.

    """

    check_params = dict(
        accept_sparse=False,
        accept_large_sparse=False,
        dtype=dtype,
        order=order,
        copy=copy,
        force_all_finite=False,
        ensure_2d=ensure_2d,
        allow_nd=allow_3d or allow_nd,
        ensure_min_samples=ensure_min_samples,
        ensure_min_features=0,
        estimator=estimator,
        input_name=input_name,
    )
    if force_all_finite not in (True, False, "allow-nan"):
        raise ValueError(
            'force_all_finite should be a bool or "allow-nan".'
            f"Got {force_all_finite} instead"
        )

    array = sklearn_check_array(array, **check_params)
    estimator_name = _check_estimator_name(estimator)
    if not (allow_nd or allow_3d) and array.ndim >= 3:
        raise ValueError(
            "Found array with dim %d. %s expected <= 2." % (array.ndim, estimator_name)
        )

    if not allow_nd and allow_3d and array.ndim >= 4:
        raise ValueError(
            "Found array with dim %d. %s expected <= 3." % (array.ndim, estimator_name)
        )

    if ravel_1d:
        if array.ndim == 1:
            return array.ravel(order=order)
        elif array.ndim == 2 and array.shape[1] == 1:
            return array.ravel(order=order)
        else:
            raise ValueError(
                "Found array with dim %d.%s expect 1 dim or column vector"
                % (array.ndim, estimator_name)
            )

    context = " by %s" % estimator_name if estimator is not None else ""
    if ensure_min_dims > 0 and (array.ndim == 2 or array.ndim == 3):
        if array.ndim == 3:
            n_dims = array.shape[1]
        else:
            n_dims = 1

        if n_dims < ensure_min_dims:
            raise ValueError(
                "Found array with %d dimension(s) (shape=%s) while a"
                " minimum of %d is required%s."
                % (n_dims, array.shape, ensure_min_dims, context)
            )

    if ensure_min_timesteps > 0 and (array.ndim == 2 or array.ndim == 3):
        n_timesteps = array.shape[-1]
        if n_timesteps < ensure_min_timesteps:
            # TODO: ignore sklearn tests for this error message and create our own
            #       estimator check
            raise ValueError(
                "Found array with %d feature(s) (shape=%s) while"
                " a minimum of %d is required%s."
                % (n_timesteps, array.shape, ensure_min_timesteps, context)
            )

    if np.issubdtype(array.dtype, np.double):
        padded_input_name = input_name + " " if input_name else ""
        if force_all_finite and not allow_eos and wb.iseos(array).any():
            raise ValueError(
                f"Input {padded_input_name}expected time series of equal length."
            )

        if force_all_finite is True and np.isnan(array).any():
            raise ValueError(f"Input {padded_input_name}contains NaN.")

        if force_all_finite and np.isposinf(array).any():
            raise ValueError(f"Input {padded_input_name}contains infinity.")

    return array
