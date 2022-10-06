# Authors: Isak Samsten
# License: BSD 3 clause

import math
import numbers

import numpy as np
from sklearn.utils.validation import _is_arraylike, check_scalar

from ..utils.validation import check_option
from ._crocket import (
    NormalWeightSampler,
    RocketFeatureEngineer,
    ShapeletWeightSampler,
    UniformWeightSampler,
)
from .base import BaseFeatureEngineerTransform

_SAMPLING_METHOD = {
    "normal": NormalWeightSampler,
    "uniform": UniformWeightSampler,
    "shapelet": ShapeletWeightSampler,
}


class RocketTransform(BaseFeatureEngineerTransform):
    """Transform a time series using random convolution features

    Attributes
    ----------
    embedding_ : Embedding
        The underlying embedding

    References
    ----------
    Dempster, Angus, François Petitjean, and Geoffrey I. Webb.
        ROCKET: exceptionally fast and accurate time series classification using
        random convolutional kernels.
        Data Mining and Knowledge Discovery 34.5 (2020): 1454-1495.
    """

    def __init__(
        self,
        n_kernels=1000,
        *,
        sampling="normal",
        sampling_params=None,
        kernel_size=None,
        bias_prob=1.0,
        normalize_prob=1.0,
        padding_prob=0.5,
        n_jobs=None,
        random_state=None,
    ):
        """
        Parameters
        ----------
        n_kernels : int, optional
            The number of kernels.

        n_jobs : int, optional
            The number of jobs to run in parallel. None means 1 and
            -1 means using all processors.

        random_state : int or RandomState, optional
            The psuodo-random number generator.
        """
        super().__init__(n_jobs=n_jobs)
        self.sampling = sampling
        self.sampling_params = sampling_params
        self.kernel_size = kernel_size
        self.bias_prob = bias_prob
        self.normalize_prob = normalize_prob
        self.padding_prob = padding_prob
        self.n_kernels = n_kernels
        self.random_state = random_state

    def _get_feature_engineer(self):
        if self.kernel_size is None:
            kernel_size = [7, 11, 13]
        elif isinstance(self.kernel_size, tuple) and len(self.kernel_size) == 2:
            min_size, max_size = self.kernel_size
            check_scalar(
                max_size,
                "max_size (kernel_size[1])",
                numbers.Real,
                min_val=min_size,
                max_val=1,
            )
            check_scalar(
                min_size,
                "min_size (kernel_size[0])",
                numbers.Real,
                min_val=0,
                max_val=max_size,
            )
            max_size = math.ceil(self.n_timesteps_in_ * max_size)
            min_size = math.ceil(self.n_timesteps_in_ * min_size)
            if min_size < 2:
                # TODO(1.2): Break backward
                if self.n_timesteps_in_ < 2:
                    min_size = 1
                else:
                    min_size = 2
            kernel_size = np.arange(min_size, max_size)
        elif _is_arraylike(self.kernel_size):
            kernel_size = self.kernel_size
        else:
            raise TypeError(
                "kernel_size must be array-like, got %s"
                % type(self.kernel_size).__qualname__
            )

        weight_sampler = check_option(_SAMPLING_METHOD, self.sampling, "sampling")(
            **({} if self.sampling_params is None else self.sampling_params)
        )
        return RocketFeatureEngineer(
            check_scalar(self.n_kernels, "n_kernels", numbers.Integral, min_val=1),
            weight_sampler,
            np.array(kernel_size, dtype=int),
            check_scalar(
                self.bias_prob, "bias_prob", numbers.Real, min_val=0, max_val=1
            ),
            check_scalar(
                self.padding_prob, "padding_prob", numbers.Real, min_val=0, max_val=1
            ),
            check_scalar(
                self.normalize_prob, "bias_prob", numbers.Real, min_val=0, max_val=1
            ),
        )


#
# class ShacketEmbedding(BaseEmbedding):
#     def __init__(
#         self,
#         n_kernels=1000,
#         *,
#         size=None,
#         standardize=True,
#         use_bias=False,
#         n_jobs=None,
#         random_state=None
#     ):
#         super().__init__(random_state=random_state, n_jobs=n_jobs)
#         self.n_kernels = n_kernels
#         self.standardize = standardize
#         self.use_bias = use_bias
#         self.size = size
#
#     def _get_feature_engineer(self):
#         if self.size is None:
#             size = [7, 11, 13]
#         elif isinstance(self.size, tuple) and len(self.size) == 2:
#             min_size, max_size = self.size
#             if min_size < 0 or min_size > max_size:
#                 raise ValueError(
#                     "`min_size` {0} <= 0 or {0} > {1}".format(min_size, max_size)
#                 )
#             if max_size > 1:
#                 raise ValueError("`max_size` {0} > 1".format(max_size))
#             max_size = int(self.n_timesteps_in_ * max_size)
#             min_size = int(self.n_timesteps_in_ * min_size)
#             if min_size < 2:
#                 min_size = 2
#             size = np.arange(min_size, max_size)
#         else:
#             size = self.size
#         return ShacketFeatureEngineer(
#             int(self.n_kernels),
#             np.array(size, dtype=int),
#             bool(self.use_bias),
#             bool(self.standardize),
#         )
