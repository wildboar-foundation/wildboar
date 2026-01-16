import heapq

import numpy as np
from scipy.sparse import csr_array
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..base import BaseEstimator
from ..distance import argmin_distance


class SegmenterMixin:
    _estimator_type = "segmenter"

    def fit_predict(self, X, y=None):
        if y is None:
            return self.fit(X).predict(X)
        else:
            return self.fit(X, y).predict(X)


class BaseSegmenter(TransformerMixin, BaseEstimator):
    """
    Base class for segmenters.

    Inheriting classes must set the ``labels_`` attribute to a list of lists
    with the index position of the segments.

    Parameters
    ----------
    metric : str or callable, optional
        The distance metric

        See ``_METRICS.keys()`` for a list of supported metrics.
    metric_params : dict, optional
        Parameters to the metric.

        Read more about the parameters in the
        :ref:`User guide <list_of_metrics>`.

    Attributes
    ----------
    labels_ : list of shape (n_samples, )
        A list of n_samples lists with the start index of the segment.
    """

    def __init__(self, metric="euclidean", metric_params=None):
        self.metric = metric
        self.metric_params = metric_params

    def predict(self, X):
        """
        Predict the position with the change point.

        The predicted segmentation is based on the  closest sample from the
        training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps)
            The input data.

        Returns
        -------
        csr_array of shape (n_samples, n_timesteps)
            A boolean array with the start of the change point set to True.
        """
        X = self._validate_data(X, reset=False)
        check_is_fitted(self)
        rowind = []
        colind = []
        data = []

        for i in range(X.shape[0]):
            closest = argmin_distance(
                X[i], self._fit_X, metric=self.metric, metric_params=self.metric_params
            )
            labels = self.labels_[closest[0, 0]]

            for start in labels:
                rowind.append(i)
                colind.append(start)
                data.append(1)

        return csr_array(
            (data, (rowind, colind)),
            shape=(X.shape[0], self.n_timesteps_in_),
            dtype=bool,
        )

    def transform(self, X):
        """
        Transform X such that each segment is labeled with a unique label.

        The predicted segmentation is based on the  closest sample from the
        training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps)
            The input data.

        Returns
        -------
        ndarray of shape (n_samples, n_timesteps)
            An array with the segments annotated with a label.
        """
        X = self._validate_data(X, reset=False)
        check_is_fitted(self)
        X_out = np.zeros((X.shape[0], self.n_timesteps_in_), dtype=float)

        for i in range(X.shape[0]):
            current_label = 1
            closest = argmin_distance(
                X[i], self._fit_X, metric=self.metric, metric_params=self.metric_params
            )
            labels = self.labels_[closest[0, 0]]

            for start in labels:
                X_out[i, start:] = current_label
                current_label += 1

        return X_out


def detect_changepoints(
    x,
    method="pelt",
    n_changepoints=None,
    min_segment_size=5,
    penalty=None,
):
    """
    Detect change points signal where behavior changes.

    Parameters
    ----------
    x : ndarray of shape (n_timesteps,)
        The time series
    method : str, default="pelt"
        Change point detection method:
        - "pelt": Pruned Exact Linear Time (automatic number of changepoints)
        - "binseg": Binary segmentation (requires n_changepoints)
    n_changepoints : int, optional
        Number of changepoints to find. Required for "binseg".
        For "pelt", this is ignored (automatic detection).
    min_segment_size : int, default=5
        Minimum size of segments between changepoints.
    penalty : float, optional
        Penalty for adding changepoints (for "pelt" method).
        Higher values = fewer changepoints. If None, uses BIC.

    Returns
    -------
    changepoints : ndarray
        Indices where changepoints occur (sorted).
    """
    n = len(x)

    if method == "pelt":
        changepoints = _detect_pelt(x, min_segment_size, penalty)
    elif method == "binseg":
        if n_changepoints is None:
            n_changepoints = max(1, n // 20)
        changepoints = _detect_binseg(x, n_changepoints, min_segment_size)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pelt' or 'binseg'.")

    return changepoints


def _detect_pelt(x, min_size, penalty):
    n = len(x)

    cumsum = np.zeros(n + 1)
    cumsum[1:] = np.cumsum(x)
    cumsum_sq = np.zeros(n + 1)
    cumsum_sq[1:] = np.cumsum(x**2)

    def segment_cost(start, end):
        if end - start < 2:
            return 0.0
        length = end - start
        total = cumsum[end] - cumsum[start]
        total_sq = cumsum_sq[end] - cumsum_sq[start]
        return total_sq - (total**2) / length

    if penalty is None:
        total_var = segment_cost(0, n)
        penalty = np.log(n) * total_var / n

    F = np.full(n + 1, np.inf)
    F[0] = -penalty
    last_change = np.zeros(n + 1, dtype=int)

    candidates = {0}

    for t in range(min_size, n + 1):
        best_cost = np.inf
        best_s = 0

        prune_set = set()
        for s in candidates:
            if t - s >= min_size:
                cost = F[s] + segment_cost(s, t) + penalty
                if cost < best_cost:
                    best_cost = cost
                    best_s = s
                if cost > best_cost + penalty:
                    prune_set.add(s)

        F[t] = best_cost
        last_change[t] = best_s
        candidates = (candidates - prune_set) | {t}

    changepoints = []
    t = n
    while t > 0:
        s = last_change[t]
        if s > 0:
            changepoints.append(s)
        t = s

    return np.array(sorted(changepoints))


def _detect_binseg(x, n_changepoints, min_size):
    n = len(x)
    changepoints = []

    def segment_variance(start: int, end: int) -> float:
        if end - start < 2:
            return 0.0
        return np.var(x[start:end]) * (end - start)

    def find_best_split(start, end):
        if end - start < 2 * min_size:
            return -1, 0.0

        total_var = segment_variance(start, end)
        best_gain = 0.0
        best_split = -1

        for split in range(start + min_size, end - min_size + 1):
            left_var = segment_variance(start, split)
            right_var = segment_variance(split, end)
            gain = total_var - (left_var + right_var)

            if gain > best_gain:
                best_gain = gain
                best_split = split

        return best_split, best_gain

    split, gain = find_best_split(0, n)
    if split > 0:
        heap = [(-gain, 0, n, split)]
    else:
        heap = []

    while heap and len(changepoints) < n_changepoints:
        _, start, end, split = heapq.heappop(heap)

        changepoints.append(split)

        left_split, left_gain = find_best_split(start, split)
        if left_split > 0 and left_gain > 0:
            heapq.heappush(heap, (-left_gain, start, split, left_split))

        right_split, right_gain = find_best_split(split, end)
        if right_split > 0 and right_gain > 0:
            heapq.heappush(heap, (-right_gain, split, end, right_split))

    return np.array(sorted(changepoints))
