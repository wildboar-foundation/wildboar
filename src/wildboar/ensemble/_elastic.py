import numbers

import numpy as np
from sklearn.base import ClassifierMixin, _fit_context
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted

from ..base import BaseEstimator
from ..distance._distance import _METRICS
from ..distance._multi_metric import make_parameter_grid
from ..distance._neighbors import KNeighborsClassifier
from ..utils.validation import check_classification_targets


def _make_elastic_parameter_grid(std):
    return {
        "dtw": {"min_r": 0.01, "max_r": 0.3, "num_r": 10},
        "adtw": {
            "min_r": 0.01,
            "max_r": 0.3,
            "num_r": 3,
            "min_p": 1,
            "max_p": 4,
            "num_p": 3,
        },
        "ddtw": {"min_r": 0.01, "max_r": 0.3, "num_r": 10},
        "wdtw": {"min_g": 0.01, "max_g": 0.5, "num_g": 10},
        "wddtw": {"min_g": 0.01, "max_g": 0.5, "num_g": 10},
        "lcss": {
            "min_r": 0.0,
            "max_r": 0.25,
            "num_r": 3,
            "min_epsilon": 0.2 * std,
            "max_epsilon": std,
            "num_epsilon": 3,
        },
        "erp": {"min_g": 0, "max_g": 1.0, "num_g": 10},
        "msm": {"min_c": 0.01, "max_c": 100, "num_c": 10},
        "twe": {
            "min_penalty": 1e-5,
            "max_penalty": 1.0,
            "num_penalty": 3,
            "min_stiffness": 1e-6,
            "max_stiffness": 0.1,
            "num_stiffness": 3,
        },
    }


def _make_non_elastic_parameter_grid():
    return {
        "euclidean": None,
        "minkowski": {"min_p": -5, "max_p": 5, "num_p": 10},
        "chebyshev": None,
        "manhattan": None,
        "angular": None,
        "cosine": None,
    }


class ElasticEnsembleClassifier(ClassifierMixin, BaseEstimator):
    """
    Ensemble of :class:`wildboar.distance.KNeighborsClassifier`.

    Each classifier is fitted with an optimized parameter grid
    over metric parameters.

    Parameters
    ----------
    n_neighbors : int, optional
        The number of neighbors.
    metric : {"auto", "elastic", "non_elastic", "all"} or dict, optional
        The metric specification.

        - if "auto" or "elastic", fit one classifier for each elastic distance
          as described by Lines and Bagnall (2015). We use a slightly smaller
          parameter grid.
        - if "non_elastic", fit one classifier for each non-elastic distance
          measure.
        - if "all", fit one classifier for the metrics in both "elastic" and
          "non_elastic".
        - if dict, a custom metric specification.
    n_jobs : int, optional
        The number of paralell jobs.

    Attributes
    ----------
    scores : tuple
        A tuple of metric name and cross-validation score.

    References
    ----------
    Jason Lines and Anthony Bagnall,
        Time Series Classification with Ensembles of Elastic Distance Measures,
        Data Mining and Knowledge Discovery, 29(3), 2015.

    Examples
    --------
    >>> from wildboar.datasets import load_gun_point
    >>> from wildboar.ensemble import ElasticEnsembleClassifier
    >>> X_train, X_test, y_train, y_test = load_gun_point(merge_train_test=False)
    >>> clf = ElasticEnsembleClassifier(
    ...     metric={
    ...         "dtw": {"min_r": 0.1, "max_r": 0.3},
    ...         "ddtw": {"min_r": 0.1, "max_r": 0.3},
    ...     },
    ... )
    >>> clf.fit(X_train, y_train)
    ElasticEnsembleClassifier(metric={'ddtw': {'max_r': 0.3, 'min_r': 0.1},
                                      'dtw': {'max_r': 0.3, 'min_r': 0.1}})
    >>> clf.score(X_test, y_test)
    0.9866666666666667
    """

    _parameter_constraints = {
        "n_neighbors": [Interval(numbers.Integral, 1, None, closed="left")],
        "metric": [dict, StrOptions({"all", "non_elastic", "elastic", "auto"})],
        "n_jobs": [int, None],
    }

    def __init__(self, n_neighbors=1, *, metric="auto", n_jobs=None):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_jobs = n_jobs

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, x, y):
        """
        Fit the estimator.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_timesteps) or (n_samples, n_dim, n_timesteps)
            The input samples.
        y : array-like of shape (n_samples, )
            The input labels.

        Returns
        -------
        object
            This estimator.
        """
        x, y = self._validate_data(x, y, allow_3d=True)
        check_classification_targets(y)

        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("too few labels")

        if self.metric in ["elastic", "auto"]:
            metric = _make_elastic_parameter_grid(x.std())
        elif self.metric == "non_elastic":
            metric = _make_non_elastic_parameter_grid()
        elif self.metric == "all":
            metric = {
                **_make_non_elastic_parameter_grid(),
                **_make_elastic_parameter_grid(x.std()),
            }
        else:
            metric = self.metric

        metric_param_grid = {}
        for metric_name, param_grid in metric.items():
            if metric_name not in _METRICS:
                raise ValueError(f"{metric_name} is not supported")
            metric_param_grid[metric_name] = make_parameter_grid(param_grid)

        self.estimators_ = []
        self.scores_ = []
        for metric, metric_params in metric_param_grid.items():
            gridcv = GridSearchCV(
                KNeighborsClassifier(n_neighbors=self.n_neighbors, metric=metric),
                param_grid={"metric_params": metric_params},
                cv=LeaveOneOut(),
                n_jobs=self.n_jobs,
            ).fit(x, y)
            estimator = gridcv.best_estimator_.set_params(n_jobs=self.n_jobs).fit(x, y)
            self.estimators_.append(estimator)
            self.scores_.append((metric, gridcv.best_score_))

        return self

    def predict_proba(self, x):
        """
        Compute probability estimates for the samples in x.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_timesteps) or (n_samples, n_dim, n_timesteps)
            The input time series.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            The probabilities.
        """
        check_is_fitted(self)
        x = self._validate_data(x, allow_3d=True, reset=False)
        proba = np.zeros((x.shape[0], len(self.estimators_), len(self.classes_)))
        score_sum = 0
        for i, ((_, score), estimator) in enumerate(
            zip(self.scores_, self.estimators_)
        ):
            proba[:, i] = estimator.predict_proba(x) * score
            score_sum += score
        return proba.sum(axis=1) / score_sum

    def predict(self, x):
        """
        Compute the class label for the samples in x.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_timesteps) or (n_samples, n_dim, n_timesteps)
            The input samples.

        Returns
        -------
        ndarray of shape (n_samples, )
            The class label for each sample.
        """
        proba = self.predict_proba(x)
        return np.take(self.classes_, np.argmax(proba, axis=1))
