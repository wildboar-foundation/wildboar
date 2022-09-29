# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np

from ._rocket_fast import (
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
        random_state=None
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
            if min_size < 0 or min_size > max_size:
                raise ValueError(
                    "`min_size` {0} <= 0 or {0} > {1}".format(min_size, max_size)
                )
            if max_size > 1:
                raise ValueError("`max_size` {0} > 1".format(max_size))
            max_size = int(self.n_timesteps_in_ * max_size)
            min_size = int(self.n_timesteps_in_ * min_size)
            if min_size < 2:
                min_size = 2
            kernel_size = np.arange(min_size, max_size)
        else:
            kernel_size = self.kernel_size

        if self.sampling in _SAMPLING_METHOD:
            sampling_params = (
                {} if self.sampling_params is None else self.sampling_params
            )
            weight_sampler = _SAMPLING_METHOD[self.sampling](**sampling_params)
        else:
            raise ValueError("sampling (%r) is not supported." % self.sampling)
        return RocketFeatureEngineer(
            int(self.n_kernels),
            weight_sampler,
            np.array(kernel_size, dtype=int),
            float(self.bias_prob),
            float(self.padding_prob),
            float(self.normalize_prob),
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
