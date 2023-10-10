import numbers

import numpy as np

from ..utils.validation import check_array
from ._cconv import conv1d


def convolve(X, kernel, bias=0.0, *, dilation=1, stride=1, padding=0):
    """
    Apply 1D convolution over a time series.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_timestep)
        The input.
    kernel : array-like of shape (kernel_size, )
        The kernel.
    bias : float, optional
        The bias.
    dilation : int, optional
       The spacing between kernel elements.
    stride : int, optional
        The stride of the convolving kernel.
    padding : int, optional
        Implicit padding on both sides of the input time series.

    Returns
    -------
    ndarray of shape (n_samples, output_size)
        The result of the convolution, where output_size is given by:::

            floor(
                ((X.shape[1] + 2 * padding) - (kernel.shape[0] - 1 * dilation + 1)) / stride
                + 1
            ).

    """
    if not isinstance(dilation, numbers.Integral) or dilation < 1:
        raise ValueError("dilation must be a strictly positive integer.")
    if not isinstance(dilation, numbers.Integral) or stride < 1:
        raise ValueError("stride must be a strictly positive integer.")
    if not isinstance(dilation, numbers.Integral) or padding < 0:
        raise ValueError("padding must be a positive integer.")

    X = check_array(X, allow_3d=False, ensure_ts_array=True, input_name="X")
    kernel = check_array(kernel, ensure_2d=False, order="c", input_name="kernel")

    kernel_size = (kernel.shape[0] - 1) * dilation + 1
    input_size = X.shape[-1] + 2 * padding

    if kernel_size > input_size:
        raise ValueError(f"kernel={kernel_size} larger than input_size={input_size}")

    out = np.empty(
        (X.shape[0], int(np.floor((input_size - kernel_size) / stride) + 1)),
        dtype=float,
    )

    conv1d(
        X,
        kernel,
        bias,
        dilation=int(dilation),
        padding=int(padding),
        stride=int(stride),
        out=out,
    )
    return out
