import math
from functools import partial
from numbers import Integral, Real

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.utils._param_validation import Interval

from ..base import BaseEstimator
from ..distance import matrix_profile


class MatrixProfileTransform(TransformerMixin, BaseEstimator):
    """
    Matrix profile transform.

    Transform each time series in a dataset to its MatrixProfile similarity
    self-join.

    Parameters
    ----------
    window : int or float, optional
        The subsequence size, by default 0.1.

        - if float, a fraction of n_timestep.
        - if int, the exact subsequence size.
    exclude : int or float, optional
        The size of the exclusion zone. The default exclusion zone is 0.2.

        - if float, expressed as a fraction of the windows size.
        - if int, exact size (0 < exclude).
    n_jobs : int, optional
        The number of jobs to use when computing the profile.

    Examples
    --------
    >>> from wildboar.datasets import load_two_lead_ecg()
    >>> from wildboar.transform import MatrixProfileTransform
    >>> x, y = load_two_lead_ecg()
    >>> t = MatrixProfileTransform()
    >>> t.fit_transform(x)
    """

    _parameter_constraints: dict = {
        "window": [
            Interval(Integral, 1, None, closed="left"),
            Interval(Real, 0, 1, closed="right"),
        ],
        "exclude": [
            None,
            Interval(Integral, 0, None, closed="left"),
            Interval(Real, 0, 1, closed="both"),
        ],
        "n_jobs": [None, Integral],
    }

    def __init__(self, window=0.1, exclude=None, n_jobs=None):
        self.window = window
        self.n_jobs = n_jobs
        self.exclude = exclude

    def fit(self, x, y=None):
        """
        Fit the matrix profile.

        Sets the expected input dimensions.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_timesteps) \
        or (n_samples, n_dims, n_timesteps)
            The samples.
        y : ignored
            The optional labels.

        Returns
        -------
        self 
            A fitted instance.
        """
        self._validate_params()
        self._validate_data(x, dtype=float, allow_3d=True)

        if self.window > self.n_timesteps_in_:
            raise ValueError(
                f"The window parameter of {type(self).__name__} must be <= X.shape[-1]"
            )
        return self

    def transform(self, x):
        """
        Transform the samples to their MatrixProfile self-join.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_timesteps) \
        or (n_samples, n_dims, n_timesteps)
            The samples.

        Returns
        -------
        ndarray of shape (n_samples, n_timestep) or (n_samples, n_dims, n_timesteps)
            The matrix matrix profile of each sample.
        """
        x = self._validate_data(x, reset=False, allow_3d=True, dtype=float)

        if isinstance(self.window, Integral):
            window = self.window
        else:
            window = math.ceil(self.window * self.n_timesteps_in_)

        func = partial(
            matrix_profile, window=window, exclude=self.exclude, n_jobs=self.n_jobs
        )
        if self.n_dims_in_ > 1:
            profile_size = self.n_timesteps_in_ - window + 1
            mp = np.empty((x.shape[0], self.n_dims_in_, profile_size), dtype=float)

            for i in range(self.n_dims_in_):
                mp[:, i, :] = func(x, dim=i)

            return mp
        else:
            return func(x)

    def _more_tags(self):
        return {"X_types": ["2darray", "3darray"]}
