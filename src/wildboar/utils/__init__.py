import numpy as np
from sklearn.utils.validation import check_array as sklearn_check_array
from wildboar.utils.data import check_dataset

__all__ = [
    "check_array",
    "check_dataset",
]


def check_array(
    x,
    allow_multivariate=False,
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
        order = kwargs.get("order")
        if order is not None and order.lower() != "c":
            raise ValueError("order=%r and contiguous=True are incompatible")
        kwargs["order"] = "C"

    if allow_multivariate:
        if kwargs.pop("ensure_2d", False) == True:
            raise ValueError(
                "ensure_2d=True and allow_multivariate=True are incompatible"
            )

        if kwargs.pop("allow_nd", True) == False:
            raise ValueError(
                "allow_nd=False and allow_multivaraite=True are incompatible"
            )
        x = sklearn_check_array(x, ensure_2d=False, allow_nd=True, **kwargs)
        if x.ndim == 0:
            raise ValueError(
                "Expected 2D or 3D array, got scalar array instead:\narray={}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single feature or array.reshape(1, -1) "
                "if it contains a single sample.".format(x)
            )
        if x.ndim == 1:
            raise ValueError(
                "Expected 2D or 3D array, got 1D array instead:\narray={}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single feature or array.reshape(1, -1) "
                "if it contains a single sample.".format(x)
            )
        if x.ndim > 3:
            raise ValueError(
                "Expected 2D or 3D array, got {}D array instead:\narray={}.\n".format(
                    x.ndim, x
                )
            )
        return x
    else:
        return sklearn_check_array(x, **kwargs)
