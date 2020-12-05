import abc

from sklearn.base import BaseEstimator


class BaseCounterfactual(BaseEstimator):
    """Base estimator for counterfactual explanations"""

    @abc.abstractmethod
    def fit(self, estimator):
        """Fit the counterfactual to a given estimator

        Parameters
        ----------
        estimator : object
            An estimator for which counterfactual explanations are produced

        Returns
        -------
        self
        """
        pass

    @abc.abstractmethod
    def transform(self, x, y):
        """Transform the i:th sample in x to a sample that would be labeled as the i:th label in y

        Parameters
        ----------
        x : array-like of shape (n_samples, n_timestep) or (n_samples, n_dimension, n_timestep)
            The samples to generate counterfactual explanations for

        y : array-like of shape (n_samples,)
            The desired label of the counterfactual sample

        Returns
        -------

        counterfactuals : ndarray of same shape as x
            The counterfactual for each sample

        success : ndarray of shape (n_samples,)
             Boolean matrix of successful transformations
        """
        pass
