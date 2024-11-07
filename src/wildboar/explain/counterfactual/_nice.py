from copy import deepcopy

import numpy as np
from sklearn.base import _fit_context
from sklearn.utils._param_validation import StrOptions

from ...base import (
    BaseEstimator,
    CounterfactualMixin,
    ExplainerMixin,
)
from ...distance import argmin_distance, pairwise_distance
from ...distance._distance import _METRICS


def _compactness_reward(
    _original,
    _current_candidate,
    current_prediction,
    _new_candidates,
    new_candidates_prediction,
):
    """
    Calculate the compactness reward for a new set of candidates.

    This function computes the difference between the predictions of new candidates
    and the current prediction. It is used to evaluate the compactness of new
    candidate solutions in comparison to the current candidate.

    Parameters
    ----------
    _original : ignored
        Ignored.
    _current_candidate : ignored
        Ignored.
    current_prediction : float
        The prediction for the current candidate.
    _new_candidates : ignored
        Ignored.
    new_candidates_prediction : ndarray of shape (n_candidates, )
        The prediction for the new candidates.

    Returns
    -------
    float
        The calculated compactness reward.
    """
    return new_candidates_prediction - current_prediction


def _proximity_reward(
    original,
    current_candidate,
    current_prediction,
    new_candidates,
    new_candidates_prediction,
):
    compactness_reward = new_candidates_prediction - current_prediction
    distance_reward = pairwise_distance(original, new_candidates) - pairwise_distance(
        original, current_candidate
    )
    """
    Calculate the proximity reward for a given candidate solution.

    The proximity reward is a measure that combines the compactness of the new
    candidates' predictions relative to the current prediction and the distance
    of the new candidates from the original instance.

    Parameters
    ----------
    original : array-like
        The original instance.
    current_candidate : array-like
        The current candidate solution.
    current_prediction : float
        The prediction for the current candidate.
    new_candidates : array-like of shape (n_candidates, n_timestep)
        The new candidate solutions.
    new_candidates_prediction : ndarray of shape (n_candidates, )
        The prediction for the new candidates.

    Returns
    -------
    float
        The calculated proximity reward.
    """
    return compactness_reward / (distance_reward + 0.0000001)


def _best_first_search(
    estimator, reward, class_indices, X, nearest_neighbors, verbose=0
):
    """
    Perform a best-first search to find counterfactual examples.

    Parameters
    ----------
    estimator : object
        A fitted estimator with a `predict_proba` method.
    reward : callable
        A function to evaluate the quality of counterfactual candidates.
    class_indices : array-like of shape (n_samples,)
        Target class indices for each sample.
    X : array-like of shape (n_samples, n_timestep)
        Original input samples.
    nearest_neighbors : array-like of shape (n_samples, n_timestep)
        Nearest neighbors for each sample in `X`.

    Returns
    -------
    ndarray of shape (n_samples, n_timestep)
        Counterfactual examples for the input samples.
    """
    counterfactuals = X.copy()
    predictions = estimator.predict_proba(X)

    # Samples that needs to change for for the prediction to change into target
    pending_idxs = set(np.where(predictions.argmax(axis=1) != class_indices)[0])

    while pending_idxs:
        completed_idx = set()
        for sample_idx in pending_idxs:
            difference_idx = ~np.isclose(
                counterfactuals[sample_idx], nearest_neighbors[sample_idx]
            )
            difference_idx = np.where(difference_idx)[0]

            if len(difference_idx) > 0:
                class_index = class_indices[sample_idx]
                candidates = np.tile(
                    counterfactuals[sample_idx], (len(difference_idx), 1)
                )

                # Create candidates with each of the possible changes.
                for candidate_index, timestep in enumerate(difference_idx):
                    candidates[candidate_index, timestep] = nearest_neighbors[
                        sample_idx, timestep
                    ]

                candidate_predictions = estimator.predict_proba(candidates)
                candidate_scores = reward(
                    X[sample_idx],
                    counterfactuals[sample_idx],
                    predictions[sample_idx, class_index],
                    candidates,
                    candidate_predictions[:, class_index],
                )

                best_candidate_idx = np.argmax(candidate_scores)
                counterfactuals[sample_idx] = candidates[best_candidate_idx]
                predictions[sample_idx] = candidate_predictions[best_candidate_idx]
                if predictions[sample_idx, class_index] > 0.5:
                    if verbose > 0:
                        print("Counterfactaul for {sample_idx} complete...")
                    completed_idx.add(sample_idx)
            else:
                if verbose > 0:
                    print("Counterfactaul for {sample_idx} failed...")
                completed_idx.add(sample_idx)

        pending_idxs.difference_update(completed_idx)

    return counterfactuals


_REWARD = {
    "compactness": _compactness_reward,
    "proximity": _proximity_reward,
}


class NiceCounterfactual(CounterfactualMixin, ExplainerMixin, BaseEstimator):
    """
    An algorithm designed to generate counterfactual explanations.

    As described by Brughmans (2024), it is designed for tabular data,
    addressing key requirements for real-life deployments:

    1. Provides explanations for all predictions.
    2. Compatible with any classification model, including non-differentiable ones.
    3. Efficient in runtime.
    4. Offers multiple counterfactual explanations with varying characteristics.

    The algorithm leverages information from a nearest unlike neighbor to
    iteratively incorporating timesteps from this neighbor into the instance
    being explained.

    Parameters
    ----------
    reward : str or callable, optional
        The reward function to optimize the counterfactual explanations. Can be
        a string specifying one of the predefined reward functions or a custom
        callable. The callable is a function `f(original, current,
        current_pred, candidates, candidate_preds)` that returns a ndarray of
        scores for each candidate.

    metric : str, optional
        The distance metric to use for calculating proximity between instances.
        Must be one of the supported metrics.

    metric_params : dict, optional
        Additional parameters to pass to the distance metric function.

    verbose : int, optional
        Increase feedback. No feedback (0) and some feedback (1).
    """

    _parameter_constraints: dict = {
        "metric": [StrOptions(_METRICS.keys())],
        "metric_params": [None, dict],
        "reward": [StrOptions(_REWARD.keys()), callable],
        "verbose": [int],
    }

    def __init__(
        self,
        reward="compactness",
        metric="euclidean",
        metric_params=None,
        verbose=0,
    ):
        self.reward = reward
        self.metric = metric
        self.metric_params = metric_params
        self.verbose = verbose

    def _validate_estimator(self, estimator):
        if not hasattr(estimator, "predict_proba"):
            raise ValueError("estimator must support predict_proba")

        return super()._validate_estimator(estimator)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, estimator, X, y):
        """
        Fit the counterfactual explanation model.

        Parameters
        ----------
        estimator : object
            The estimator object to be validated and used for fitting.
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self
            Returns the instance itself.
        """
        self.estimator_ = deepcopy(self._validate_estimator(estimator))
        X, y = self._validate_data(X, y, reset=False)
        # mask = estimator.predict(X) == y
        # TODO: mask can be used to 'justify' CFs

        if isinstance(self.reward, str):
            self.reward_ = _REWARD[self.reward]
        elif hasattr(self.reward, "__call__"):
            self.reward_ = self.reward

        self.neighbors_ = {c: X[y == c] for c in self.classes_}

        return self

    def explain(self, X, y):
        """
        Explain the predictions for the given data.

        Parameters
        ----------
        X : array-like
            The input data for which explanations are to be generated.
        y : array-like
            The target values corresponding to the input data.
        """
        X, y = self._validate_data(X, y, reset=False)
        class_indices = np.searchsorted(
            self.classes_, y, sorter=np.argsort(self.classes_)
        )

        nearest_neighbors = np.empty_like(X)
        for current_class in np.unique(y):
            current_class_mask = y == current_class
            neighbor_candidates = self.neighbors_[current_class]

            # Find the closest neighbor among the desired class
            neighbor_candidate_index = argmin_distance(
                X[current_class_mask],
                neighbor_candidates,
                k=1,
                metric=self.metric,
                metric_params=self.metric_params,
            )[:, 0]
            nearest_neighbors[current_class_mask] = neighbor_candidates[
                neighbor_candidate_index
            ]

        return _best_first_search(
            self.estimator_,
            self.reward_,
            class_indices,
            X,
            nearest_neighbors,
            self.verbose,
        )
