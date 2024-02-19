import numbers

import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.utils._param_validation import Interval, StrOptions

from ._base import BaseDistanceSelector


class _UnivariateDistanceSelector(BaseDistanceSelector):
    _parameter_constraints = {
        **BaseDistanceSelector._parameter_constraints,
        "score_func": [callable],
    }

    def __init__(self, score_func, *, metric, metric_params, n_jobs):
        super().__init__(metric=metric, metric_params=metric_params, n_jobs=n_jobs)
        self.score_func = score_func

    def _check_params(self):
        pass

    def _fit(self, distance, y):
        self._check_params()
        self.scores_ = np.empty(self.n_dims_in_, dtype=float)
        self.pvalues_ = None
        for i in range(distance.shape[0]):
            score_result = self.score_func(distance[i], y)
            if isinstance(score_result, (list, tuple)):
                score, pvalue = map(np.asarray, score_result)
                self.scores_[i] = score.mean()
                if self.pvalues_ is None:
                    self.pvalues_ = np.empty(self.n_dims_in_, dtype=float)

                self.pvalues_[i] = pvalue.mean()
            else:
                score = np.asarray(score_result)
                self.scores_[i] = score.mean()


class SelectDimensionPercentile(_UnivariateDistanceSelector):
    """
    Select the fraction of dimensions with largest score.

    For each dimension, the pairwise distance between time series is computed
    and dimensions with the lowest scores are removed.

    Parameters
    ----------
    score_func : callable, optional
        Function taking two arrays X and y and returning scores and optionally
        p-values. The default is :func:`~sklearn.feature_selection.f_classif`.
    percentile : float, optional
        Percent of dimensions to retain.
    metric : str, optional
        The distance metric.
    metric_params : dict, optional
        Optional parameters to the distance metric.

        Read more about the metrics and their parameters in the
        :ref:`User guide <list_of_metrics>`.
    n_jobs : int, optional
        The number of parallel jobs.

    Examples
    --------
    >>> from wildboar.datasets import load_ering
    >>> from wildboar.dimension_selection import SelectDimensionPercentile
    >>> X, y = load_ering()
    >>> sdp = SelectDimensionPercentile(percentile=50)
    >>> sdp.fit(X, y)
    SelectDimensionPercentile(percentile=50)
    >>> sdp.get_dimensions()
    array([False, False,  True,  True])
    >>> sdp.transform(X).shape
    """

    _parameter_constraints = {
        **BaseDistanceSelector._parameter_constraints,
        **_UnivariateDistanceSelector._parameter_constraints,
        "percentile": [Interval(numbers.Real, 0, 100, closed="both")],
    }

    def __init__(
        self,
        score_func=f_classif,
        *,
        percentile=10,
        metric="euclidean",
        metric_params=None,
        n_jobs=None,
    ):
        super().__init__(
            score_func, metric=metric, metric_params=metric_params, n_jobs=n_jobs
        )
        self.percentile = percentile

    def _get_dimensions(self):
        if self.percentile == 100:
            return np.ones(self.n_dims_in_, dtype=bool)
        elif self.percentile == 0:
            return np.zeros(self.n_dims_in_, dtype=bool)

        scores = self.scores_.copy()
        scores[np.isnan(scores)] = -np.inf
        threshold = np.percentile(scores, 100 - self.percentile)
        ties = np.flatnonzero(threshold == scores)
        mask = scores > threshold
        if ties.size > 0:
            keep = np.floor(self.n_dims_in_ * self.percentile / 100) - mask.sum()
            mask[ties[:keep]] = True

        return mask


class SelectDimensionTopK(_UnivariateDistanceSelector):
    """
    Select the dimensions with the `k` highest scores.

    For each dimension, the pairwise distance between time series is computed
    and dimensions with the lowest scores are removed.

    Parameters
    ----------
    score_func : callable, optional
        Function taking two arrays X and y and returning scores and optionally
        p-values. The default is :func:`~sklearn.feature_selection.f_classif`.
    k : int, optional
        The number of top dimensions to select.
    metric : str, optional
        The distance metric.
    metric_params : dict, optional
        Optional parameters to the distance metric.

        Read more about the metrics and their parameters in the
        :ref:`User guide <list_of_metrics>`.
    n_jobs : int, optional
        The number of parallel jobs.

    Examples
    --------
    >>> from wildboar.datasets import load_ering
    >>> from wildboar.dimension_selection import SelectDimensionTopK
    >>> X, y = load_ering()
    >>> sdt = SelectDimensionTopK(k=1)
    >>> sdt.fit(X, y)
    SelectDimensionTopK(k=1)
    >>> sdt.get_dimensions()
    array([False, False,  True, False])
    >>> sdt.transform(X).shape
    (300, 65)
    """

    _parameter_constraints = {
        **BaseDistanceSelector._parameter_constraints,
        **_UnivariateDistanceSelector._parameter_constraints,
        "k": [Interval(numbers.Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        score_func=f_classif,
        *,
        k=None,
        metric="euclidean",
        metric_params=None,
        n_jobs=None,
    ):
        super().__init__(
            score_func, metric=metric, metric_params=metric_params, n_jobs=n_jobs
        )
        self.k = k

    def _check_params(self):
        if self.k is not None and self.k > self.n_dims_in_:
            raise ValueError("k must be smaller than the number of dimensions")

    def _get_dimensions(self):
        mask = np.zeros(self.n_dims_in_, dtype=bool)
        argsort = np.argsort(self.scores_, kind="stable")
        mask[argsort[-self.k :]] = True
        return mask


class SelectDimensionSignificance(_UnivariateDistanceSelector):
    """
    Select dimensions with a p-value below `alpha`.

    For each dimension, the pairwise distance between time series is computed
    and dimensions with p-values above `alpha` is removed.

    Parameters
    ----------
    score_func : callable, optional
        Function taking two arrays X and y and returning scores and optionally
        p-values. The default is :func:`~sklearn.feature_selection.f_classif`.
    alpha : int, optional
        Percent of dimensions to retain.
    method : {"fpr", "fdr", "fwe"}, optional
        The method for correcting the alpha value.

        - If `"fpr"`, false positive rate, apply no correction.
        - If `"fdr"`, false discovery rate, apply the Benjamini-Hochberg procedure.
        - If `"fwer"`, family-wise error rate, apply the Bonferroni procedure.
    metric : str, optional
        The distance metric.
    metric_params : dict, optional
        Optional parameters to the distance metric.

        Read more about the metrics and their parameters in the
        :ref:`User guide <list_of_metrics>`.
    n_jobs : int, optional
        The number of parallel jobs.

    Examples
    --------
    >>> from wildboar.datasets import load_basic_motions
    >>> from wildboar.dimension_selection import SelectDimensionSignificance
    >>> X, y = load_basic_motions()
    >>> sds = SelectDimensionSignificance(alpha=0.01)
    >>> sds.fit(X, y)
    SelectDimensionSignificance(alpha=0.01)
    >>> sds.get_dimensions()
    array([ True,  True,  True,  True,  True,  True])
    >>> sds.transform(X).shape
    (80, 6, 100)
    """

    _parameter_constraints = {
        **BaseDistanceSelector._parameter_constraints,
        **_UnivariateDistanceSelector._parameter_constraints,
        "alpha": [Interval(numbers.Real, 0, None, closed="neither")],
        "method": [StrOptions({"fpr", "fdr", "fwer"})],
    }

    def __init__(
        self,
        score_func=f_classif,
        *,
        alpha=0.05,
        method="fpr",
        metric="euclidean",
        metric_params=None,
        n_jobs=None,
    ):
        super().__init__(
            score_func, metric=metric, metric_params=metric_params, n_jobs=n_jobs
        )
        self.alpha = alpha
        self.method = method

    def _get_dimensions(self):
        if self.pvalues_ is None:
            raise ValueError("score_func does not provide p-values")

        if self.method == "fpr":
            return self.pvalues_ < self.alpha
        elif self.method == "fdr":
            sorted_pvalues = np.sort(self.pvalues_, kind="stable")
            alphas = self.alpha / np.arange(
                self.n_dims_in_, self.n_dims_in_ * self.n_dims_in_ + 1, self.n_dims_in_
            )
            selected_pvalues = sorted_pvalues[sorted_pvalues <= alphas]
            if len(selected_pvalues) == 0:
                return np.zeros(self.n_dims_in_, dtype=bool)
            else:
                return self.pvalues_ <= selected_pvalues.max()
        elif self.method == "fwer":
            return self.pvalues_ < self.alpha / self.n_dims_in_
        else:
            raise ValueError(
                f"method must be 'fpr', 'fdr' or 'fwer', got {self.method}"
            )
