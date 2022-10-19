import math
import numbers
from functools import partial

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_scalar

from ..base import BaseEstimator
from ..distance import matrix_profile


class MatrixProfileTransform(TransformerMixin, BaseEstimator):
    """Transform each time series in a dataset to its MatrixProfile similarity self-join

    Examples
    --------
    >>> from wildboar.datasets import load_two_lead_ecg()
    >>> from wildboar.transform import MatrixProfileTransform
    >>> x, y = load_two_lead_ecg()
    >>> t = MatrixProfileTransform()
    >>> t.fit_transform(x)

    """

    def __init__(self, window=0.1, exclude=None, n_jobs=None):
        """
        Parameters
        ----------
        window : int or float, optional
        The subsequence size, by default 0.1

        - if float, a fraction of n_timestep
        - if int, the exact subsequence size

        exclude : int or float, optional
            The size of the exclusion zone. The default exclusion zone is 0.2

            - if float, expressed as a fraction of the windows size
            - if int, exact size (0 < exclude)

        n_jobs : int, optional
            The number of jobs to use when computing the
        """
        self.window = window
        self.n_jobs = n_jobs
        self.exclude = exclude

    def fit(self, x, y=None):
        """Fit the matrix profile. Sets the expected input dimensions

        Parameters
        ----------
        x : array-like of shape (n_samples, n_timesteps) \
        or (n_samples, n_dims, n_timesteps)
            The samples

        y : ignored
            The optional labels.

        Returns
        -------
        self : a fitted instance
        """
        self._validate_data(x, dtype=float, allow_3d=True)
        return self

    def transform(self, x):
        """Transform the samples to their MatrixProfile self-join.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_timesteps) \
        or (n_samples, n_dims, n_timesteps)
            The samples

        Returns
        -------
        mp : ndarray of shape (n_samples, n_timestep) \
        or (n_samples, n_dims, n_timesteps)
            The matrix matrix profile of each sample
        """
        x = self._validate_data(x, reset=False, allow_3d=True, dtype=float)

        if isinstance(self.window, numbers.Integral):
            window = check_scalar(
                self.window,
                "window",
                numbers.Integral,
                min_val=1,
                max_val=self.n_timesteps_in_,
            )
        elif isinstance(self.window, numbers.Real):
            window = math.ceil(
                check_scalar(
                    self.window,
                    "window",
                    numbers.Real,
                    min_val=0,
                    max_val=1,
                    include_boundaries="right",
                )
                * self.n_timesteps_in_
            )
        else:
            raise TypeError(
                "window must be int or float, not %r" % type(self.window).__qualname__
            )

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
