# Authors: Isak Samsten
# License: BSD 3 clause

import numbers

import numpy as np
from sklearn.utils._param_validation import Interval, StrOptions

from ._base import BaseFeatureEngineerTransform
from ._crocket import (
    NormalWeightSampler,
    RocketFeatureEngineer,
    ShapeletWeightSampler,
    UniformWeightSampler,
)

_SAMPLING_METHOD = {
    "normal": NormalWeightSampler,
    "uniform": UniformWeightSampler,
    "shapelet": ShapeletWeightSampler,
}


class RocketMixin:
    _parameter_constraints: dict = {
        "n_kernels": [
            Interval(numbers.Integral, 1, None, closed="left"),
        ],
        "kernel_size": ["array-like", None],
        "min_size": [
            Interval(numbers.Real, 0, 1, closed="both"),
            None,
        ],
        "max_size": [
            Interval(numbers.Real, 0, 1, closed="both"),
            None,
        ],
        "sampling": [StrOptions(_SAMPLING_METHOD.keys())],
        "sampling_params": [dict, None],
        "bias_prob": [Interval(numbers.Real, 0, 1, closed="both")],
        "padding_prob": [Interval(numbers.Real, 0, 1, closed="both")],
        "normalize_prob": [Interval(numbers.Real, 0, 1, closed="both")],
    }

    def _get_feature_engineer(self, n_samples):
        if self.min_size is not None or self.max_size is not None:
            if self.kernel_size is not None:
                raise ValueError(
                    f"Both min_size or max_size and kernel_size of "
                    f"{type(self).__name__} cannot be set at the same time."
                )

            min_size = self.min_size if self.min_size is not None else 0
            max_size = self.max_size if self.max_size is not None else 1

            if min_size > max_size:
                raise ValueError(
                    f"The min_size parameter of {type(self).__name__} "
                    "must be <= max_size."
                )

            max_size = int(self.n_timesteps_in_ * max_size)
            min_size = int(self.n_timesteps_in_ * min_size)
            if min_size < 2:
                min_size = 2
            if max_size < 3:
                max_size = 3

            kernel_size = np.arange(min_size, max_size)
        elif self.kernel_size is None:
            kernel_size = np.array([7, 11, 13], dtype=int)
        else:
            kernel_size = np.array(self.kernel_size, dtype=int)
            if np.min(kernel_size) < 2:
                raise ValueError("The minimum kernel size is 2.")

        WeightSampler = _SAMPLING_METHOD[self.sampling]
        sampling_params = {} if self.sampling_params is None else self.sampling_params
        return RocketFeatureEngineer(
            self.n_kernels,
            WeightSampler(**sampling_params),
            kernel_size,
            self.bias_prob,
            self.padding_prob,
            self.normalize_prob,
        )


class RocketTransform(RocketMixin, BaseFeatureEngineerTransform):
    """Transform a time series using random convolution features.

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

    _parameter_constraints: dict = {
        **RocketMixin._parameter_constraints,
        **BaseFeatureEngineerTransform._parameter_constraints,
    }

    def __init__(
        self,
        n_kernels=1000,
        *,
        sampling="normal",
        sampling_params=None,
        kernel_size=None,
        min_size=None,
        max_size=None,
        bias_prob=1.0,
        normalize_prob=1.0,
        padding_prob=0.5,
        n_jobs=None,
        random_state=None,
    ):
        """Construct a new rocket transform.

        Parameters
        ----------
        n_kernels : int, optional
            The number of kernels to sample at each node.

        sampling : {"normal", "uniform", "shapelet"}, optional
            The sampling of convolutional filters.

            - if "normal", sample filter according to a normal distribution with
              ``mean`` and ``scale``.
            - if "uniform", sample filter according to a uniform distribution with
              ``lower`` and ``upper``.
            - if "shapelet", sample filters as subsequences in the training data.
        sampling_params : dict, optional
            The parameters for the sampling.

            - if "normal", ``{"mean": float, "scale": float}``, defaults to
               ``{"mean": 0, "scale": 1}``.
            - if "uniform", ``{"lower": float, "upper": float}``, defaults to
               ``{"lower": -1, "upper": 1}``.
        kernel_size : array-like, optional
            The kernel size, by default ``[7, 11, 13]``.
        min_size : float, optional
            The minimum timestep fraction to generate kernel sizes. If set,
            ``kernel_size`` cannot be set.
        max_size : float, optional
            The maximum timestep fractio to generate kernel sizes, If set,
            ``kernel_size`` cannot be set.
        bias_prob : float, optional
            The probability of using a bias term.
        normalize_prob : float, optional
            The probability of performing normalization.
        padding_prob : float, optional
            The probability of padding with zeros.
        random_state : int or RandomState
            - If `int`, `random_state` is the seed used by the random number generator
            - If `RandomState` instance, `random_state` is the random number generator
            - If `None`, the random number generator is the `RandomState` instance used
              by `np.random`.
        """
        super().__init__(n_jobs=n_jobs, random_state=random_state)
        self.sampling = sampling
        self.sampling_params = sampling_params
        self.kernel_size = kernel_size
        self.min_size = min_size
        self.max_size = max_size
        self.bias_prob = bias_prob
        self.normalize_prob = normalize_prob
        self.padding_prob = padding_prob
        self.n_kernels = n_kernels
