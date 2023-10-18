# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: initializedcheck=False
import numpy as np
from libc.math cimport floor
from libc.stdlib cimport labs

from ..utils cimport TSArray

# TODO: Consider a fast code-path for:
#   - dilation = 1
#   - stride = 1
#   - padding = 0
cdef void convolution_1d(
    Py_ssize_t stride,
    Py_ssize_t dilation,
    Py_ssize_t padding,
    double bias,
    double *kernel,
    Py_ssize_t k_len,
    const double* x,
    Py_ssize_t x_len,
    double* out,
) noexcept nogil:
    """
    Compute the 1d convolution.

    Parameters
    ----------
    stride : int
        The stride in x.
    dilation : int
        Inflate the kernel by inserting spaces between kernel values.
    padding : int
        Increase the size of the input by padding.
    bias : double
        The bias.
    kernel : double*
        The kernel values.
    k_len : int
        The length of the kernel.
    x : double*
        The input sample.
    x_len : int
        The length of the sample.
    out : double*, output
        The output buffer.

        The invariant is not checked.

    Warnings
    --------
    This code performs no bounds checks. Ensure that the following invariants hold:

    The dilated kernel size must be smaller than the padded sample:::

        (k_len - 1) * dilation - 1 <= x_len + 2 * padding

    The output buffer must be allocated with enough memory:::

        ((x_len + 2 * padding) - (k_len - 1) * dilation + 1) / stride + 1

    """
    # Fast-path for the simple case.
    if padding == 0 and stride == 1 and dilation == 1:
        _convolution_1d_fast(bias, x, x_len, kernel, k_len, out)
        return

    cdef Py_ssize_t input_size = x_len + 2 * padding
    cdef Py_ssize_t kernel_size = (k_len - 1) * dilation + 1
    cdef Py_ssize_t output_size = <Py_ssize_t> floor((input_size - kernel_size) / stride) + 1

    cdef Py_ssize_t j  # the index in the kernel and input array
    cdef Py_ssize_t i  # the index of the output array
    cdef Py_ssize_t padding_offset
    cdef Py_ssize_t input_offset
    cdef Py_ssize_t kernel_offset
    cdef Py_ssize_t convolution_size
    cdef double inner_prod

    for i in range(output_size):
        padding_offset = padding - i * stride

        # this part of the code ensures that the iterators responsible
        # for selecting the correct values in `kernel` and `x` start
        # at the correct location. the main idea is to always start
        # the index at the first non-dilated index *after* padding.
        #
        # example:
        #  k = [1, 1, 2]
        #  d = 2
        #  k_d = [1, 0, 1, 0, 2]
        #  x = [1, 2, 3, 4, 5, 6]
        #  p = 2
        #  x_p = [0, 0, 1, 2, 3, 4, 5, 6, 0, 0]
        #
        #  first convoluton (i = 0):
        #
        #          / start the kernel index here, since p - 1 * s % d == 0,
        #          | that is: 2 % 2 == 0
        #          |
        # k_p  1 0 1 0 2
        #      | | | | |
        # x_p  0 0 1 2 3 4 5 6 0 0 (result: 7)
        #      |   |
        #      |   \ start the input index here, which is given by
        #      |     kernel_offset - padding_offset, that is 2 - 2 = 0
        #      |
        #      \ padding_offset is 2, so we are at the first padded value,
        #        but we can ignore this part since the pad is all zeros
        #        so we move head of the kernel iterator to first non-dilated
        #        value, which is located at imaginary kernel index 2, which also
        #        happens to be at the start of the input.
        #
        #  second convolution
        #
        #
        #            / since 1 % 2 != 0, we move the start of the kernel to the
        #            | the first non-dilated value, which is given by
        #            | 1 + 2 - (1 % 2) = 2
        #            |
        # k_p    1 0 1 0 2
        #        | | | | |
        # x_p  0 0 1 2 3 4 5 6 0 0 (result: 10)
        #        | |
        #        | \ we move the input iterator the the index where the
        #        |   first non dilated value is, kernel_offset (2) - padding_offset (1)
        #        \
        #         padding_offset = 1, so we are at the second padded value.
        #
        #  third convolution
        #
        #          / padding_offset = 0, so we should start the kernel
        #          | with the first value (which by definition is non-dilated).
        #          |
        # k_p      1 0 1 0 2
        #          | | | | |
        # x_p  0 0 1 2 3 4 5 6 0 0 (result: 14)
        #          |
        #          \ we should start x at the current (strided) index.
        #
        # we continue iterating, until we have visited all start locations
        # but stopping the convolution at the end of the input array.
        if padding_offset > 0:
            if padding_offset % dilation == 0:
                kernel_offset = padding_offset
            else:
                kernel_offset = padding_offset + dilation - (padding_offset % dilation)
            input_offset = kernel_offset - padding_offset
        else:
            kernel_offset = 0
            input_offset = labs(padding_offset)

        # the iteration should be performed up until but not including the
        # padding. so, the last value we should convolve over is either the
        # last value of the input, or a value before that located where the
        # kernel ends. the kernel size here takes into account dilation. we
        # express this as a length from input_offset until the end.
        convolution_size = (
            min(x_len, input_offset + kernel_size - max(0, padding_offset))
            - input_offset
        )
        inner_prod = bias
        for j from 0 <= j < convolution_size by dilation:
            # we adjust the index (j) to account for the input_offset
            # and the kernel_offset. the kernel index is in the range
            # of the dilated kernel so we take the mod to scale it back
            # to the original, non-dilated, array.
            inner_prod += x[input_offset + j] * kernel[((j + kernel_offset) // dilation)]

        out[i] = inner_prod


cdef void _convolution_1d_fast(
    double bias,
    const double *x,
    Py_ssize_t x_len,
    double *kernel,
    Py_ssize_t k_len,
    double *out
) noexcept nogil:
    cdef Py_ssize_t output_size = x_len - k_len + 1
    cdef Py_ssize_t i, j
    cdef double inner_prod
    for i in range(x_len - k_len + 1):
        inner_prod = bias
        for j in range(k_len):
            inner_prod += x[i + j] * kernel[j]
        out[i] = inner_prod


def conv1d(
    TSArray X,
    double[::1] kernel,
    double bias,
    Py_ssize_t dilation,
    Py_ssize_t padding,
    Py_ssize_t stride,
    double[:, ::1] out,
):
    cdef Py_ssize_t i
    with nogil:
        for i in range(X.shape[0]):
            convolution_1d(
                stride,
                dilation,
                padding,
                bias,
                &kernel[0],
                kernel.shape[0],
                &X[i, 0, 0],
                X.shape[2],
                &out[i, 0],
            )
