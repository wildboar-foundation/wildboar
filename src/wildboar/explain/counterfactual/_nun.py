import numbers
from abc import ABCMeta, abstractmethod
from copy import deepcopy

import numpy as np
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import (
    _is_arraylike_not_scalar,
    check_is_fitted,
    check_random_state,
)

from ...base import BaseEstimator, CounterfactualMixin, ExplainerMixin
from ...distance import argmin_distance
from ...distance._distance import _METRICS
from ...explain import IntervalImportance
from ._helper import make_target_evaluator


class _Importance(metaclass=ABCMeta):
    def fit(self, estimator, x, y):
        return self

    @abstractmethod
    def explain(self, x):
        pass


class _IntervalImportance(_Importance):
    def __init__(self, window, random_state):
        self.window = window
        self.random_state = random_state

    def fit(self, estimator, x, y):
        y_pred = estimator.predict(x)
        self.explainer_ = {
            label: IntervalImportance(
                n_repeat=1, window=self.window, random_state=self.random_state
            ).fit(estimator, x[y_pred == label], y[y_pred == label])
            for label in estimator.classes_
        }

    def explain(self, x, y):
        labels, inv = np.unique(y, return_inverse=True)
        out = np.empty_like(x)
        for i, y_idx in enumerate(inv):
            label = labels[y_idx]
            out[i] = self.explainer_[label].explain(x[i].reshape(1, -1)).reshape(-1)

        return out


class _PassthroughImportance(_Importance):
    def __init__(self, importance):
        self._importance = importance

    def explain(self, x, y):
        return np.broadcast_to(self._importance, x.shape)


class _CallableImportance(_Importance):
    def __init__(self, func):
        self._func = func

    def explain(self, x, y):
        return self._func(x, y)


class NativeGuideCounterfactual(CounterfactualMixin, ExplainerMixin, BaseEstimator):
    """
    Native guide counterfactual explanations.

    Counterfactual explanations are constructed by replacing parts of the
    explained sample with the most important region from the closest sample
    of the desired class.

    Parameters
    ----------
    metric : str or callable, optional
        The distance metric

        See ``_METRICS.keys()`` for a list of supported metrics.
    metric_params : dict, optional
        Parameters to the metric.

        Read more about the parameters in the
        :ref:`User guide <list_of_metrics>`.
    importance : {"interval"}, array-like or callable, optional
        The importance assigned to the time steps.

        - If "interval", use :class:`~wildboar.explain.IntervalImportance` to
          assign the importance of the time steps.
        - If array-like, an array of shape (n_timestep, ).
        - If callable, a function ``f(x, y)``, where `x` and `y`
          are the time series and label being explained. The return value is a
          ndarray of shape (n_timestep, ).
    target : {"predict"} or float, optional
        The target evaluation of counterfactuals:

        - if 'predict' the counterfactual prediction must return the correct
          label.
        - if float, the counterfactual prediction probability must exceed
          target value.
    window : int, optional
        The `window` parameter. Only used if `importance="interval"`.
    max_iter : int, optional
        The maximum number of iterations.
    random_state : RandomState or int, optional
        Pseudo-random number for consistency between different runs.
    n_jobs : int, optional
        The number of parallel jobs.

    Attributes
    ----------
    target_ : TargetEvaluator
        The target evaluator.
    importance_ : Importance
        The importance.
    estimator_ : Estimator
        The estimator.
    clasess_ : ndarray
        The classes known to the explainer.

    Notes
    -----
    The current implementation uses the
    :class:`~wildboar.explain.IntervalImportance` as the default method for
    assigning importances and selecting the time points where to grow the
    replacement. Unfortunately this method assigns the same score for each
    sample, that is, it provides a model level interpretation of the importance
    of each time step. To exactly replicate the work by Delaney (2021), you
    have to supply your own importance function. The default recommendation by
    the original authors is to use GradCAM.

    References
    ----------
    Delaney, E., Greene, D., Keane, M.T. (2021)
        Instance-Based Counterfactual Explanations for Time Series
        Classification. Case-Based Reasoning Research and Development, vol.
        12877, pp. 32–47. Springer International Publishing, Cham  Science.

    Examples
    --------

    >>> from wildboar.datasets import load_gun_point
    >>> from wildboar.distance import KNeighborsClassifier
    >>> from wildboar.explain.counterfactual import NativeGuideCounterfactual
    >>> X_train, X_test, y_train, y_test = load_gun_point(merge_train_test=False)
    >>> clf = KNeighborsClassifier(n_neighbors=1)
    >>> clf.fit(X_train, y_train)
    >>> ngc = NativeGuideCounterfactual(window=1, target=0.51)
    >>> ngc.fit(clf, X_train, y_train)
    >>> X_test[1:3]
    array([2., 2.], dtype=float32)
    >>> cf = nfc.explain(X_test[1:3], [1, 1])  # Desired label is [1, 1]
    >>> clf.predict(cf)
    array([1., 1.], dtype=float32)
    """  # noqa: E501

    _parameter_constraints: dict = {
        "metric": [StrOptions(_METRICS.keys()), callable],
        "metric_params": [None, dict],
        "importance": [None, StrOptions({"interval"}), "array-like", callable],
        "target": [
            StrOptions({"predict"}),
            Interval(numbers.Real, 0.0, 1, closed="right"),
        ],
        "window": [Interval(numbers.Integral, 1, None, closed="left")],
        "max_iter": [Interval(numbers.Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
        "n_jobs": [None, numbers.Integral],
    }

    def __init__(
        self,
        *,
        metric="euclidean",
        metric_params=None,
        importance="interval",
        target="predict",
        window=2,
        max_iter=100,
        random_state=None,
        n_jobs=None,
    ):
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        self.random_state = random_state
        self.importance = importance
        self.window = window
        self.target = target

    def fit(self, estimator, x, y):
        self._validate_params()
        self._validate_estimator(estimator)
        x, y = self._validate_data(x, y, reset=False, dtype=float)

        random_state = check_random_state(self.random_state)

        self.estimator_ = deepcopy(estimator)
        self.classes_ = estimator.classes_
        self._fit_X = {label: x[y == label] for label in self.classes_}
        self.target_ = make_target_evaluator(self.estimator_, self.target)

        if callable(self.importance):
            self.importance_ = _CallableImportance(self.importance)
        elif _is_arraylike_not_scalar(self.importance):
            if len(self.importance) != x.shape[-1]:
                raise ValueError(
                    "importance must be of shape (%d, ), got %d"
                    % (x.shape[-1], len(self.importance))
                )
            self.importance_ = _PassthroughImportance(self.importance)
        elif isinstance(self.importance, str) and self.importance == "interval":
            self.importance_ = _IntervalImportance(
                self.window, random_state.randint(np.iinfo(np.int32).max)
            )
        else:
            raise ValueError(
                "importance must be 'interval' or callable, got %r" % self.importance
            )

        self.importance_.fit(estimator, x, y)

    def explain(self, x, y):
        check_is_fitted(self)

        nuns = {
            label: argmin_distance(
                x,
                self._fit_X[label],
                k=1,
                metric=self.metric,
                metric_params=self.metric_params,
                n_jobs=self.n_jobs,
            )
            for label in self.classes_
        }
        importance = self.importance_.explain(x, y)
        x_cf = np.empty_like(x)
        for i in range(x.shape[0]):
            nun_idx = nuns[y[i]][i][0]
            x_cf[i] = self._explain_sample(
                x[i], y[i], self._fit_X[y[i]][nun_idx], importance[i]
            )

        return x_cf

    def _explain_sample(self, x, y, nun, importance):
        iter = 0
        window = 1
        x_tmp = np.empty_like(x)
        while iter < self.max_iter and window < self.n_timesteps_in_:
            x_tmp[:] = x
            start = _largest_magnitude_window(importance, window)
            x_tmp[start : (start + window)] = nun[start : (start + window)]
            if self.target_.is_counterfactual(x_tmp, y):
                break

            iter += 1
            window += 1

        return x_tmp


def _largest_magnitude_window(importances, window):
    return (
        np.lib.stride_tricks.sliding_window_view(importances, window)
        .sum(axis=1)
        .argmax()
    )
