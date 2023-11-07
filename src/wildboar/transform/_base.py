# Authors: Isak Samsten
# License: BSD 3 clause
from abc import ABCMeta, abstractmethod
from numbers import Integral

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_random_state

from ..base import BaseEstimator
from ..utils.validation import _check_ts_array
from ._attribute_transform import (
    fit,
    fit_transform,
    transform,
)


class BaseAttributeTransform(TransformerMixin, BaseEstimator, metaclass=ABCMeta):
    """
    Base feature engineer transform.

    Parameters
    ----------
    random_state : int or RandomState, optional
        Controls the random resampling of the original dataset.

        - If `int`, `random_state` is the seed used by the
          random number generator.
        - If :class:`numpy.random.RandomState` instance, `random_state` is the
          random number generator.
        - If `None`, the random number generator is the
          :class:`numpy.random.RandomState` instance used by
          :func:`numpy.random`.
    n_jobs : int, optional
        The number of jobs to run in parallel. A value of `None` means using a
        single core and a value of `-1` means using all cores. Positive
        integers mean the exact number of cores.

    Attributes
    ----------
    embedding_ : Embedding
        The underlying embedding.
    """

    _parameter_constraints: dict = {
        "random_state": ["random_state"],
        "n_jobs": [None, Integral],
    }

    def __init__(self, *, random_state=None, n_jobs=None):
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, x, y=None):
        """
        Fit the transform.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_timestep) or\
                (n_samples, n_dimensions, n_timestep)
            The time series dataset.
        y : None, optional
            For compatibility.

        Returns
        -------
        BaseAttributeTransform
            This object.
        """
        self._validate_params()
        x = self._validate_data(x, allow_3d=True, dtype=np.double)
        self.embedding_ = fit(
            self._get_generator(x, y),
            _check_ts_array(x),
            check_random_state(self.random_state),
        )
        return self

    def transform(self, x):
        """
        Transform the dataset.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_timestep) or\
                (n_samples, n_dimensions, n_timestep)
            The time series dataset.

        Returns
        -------
        ndarray of shape (n_samples, n_outputs)
            The transformation.
        """
        check_is_fitted(self, attributes="embedding_")
        x = self._validate_data(x, reset=False, allow_3d=True, dtype=np.double)
        return transform(self.embedding_, _check_ts_array(x), self.n_jobs)

    def fit_transform(self, x, y=None):
        """
        Fit the embedding and return the transform of x.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_timestep) or\
                (n_samples, n_dimensions, n_timestep)
            The time series dataset.
        y : None, optional
            For compatibility.

        Returns
        -------
        ndarray of shape (n_samples, n_outputs)
            The embedding.
        """
        self._validate_params()
        x = self._validate_data(x, allow_3d=True, dtype=np.double)
        embedding, x_out = fit_transform(
            self._get_generator(x, y),
            _check_ts_array(x),
            check_random_state(self.random_state),
            self.n_jobs,
        )
        self.embedding_ = embedding
        return x_out

    @abstractmethod
    def _get_generator(self, x, y):
        pass
