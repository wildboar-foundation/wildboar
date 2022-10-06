# Authors: Isak Samsten
# License: BSD 3 clause

import warnings

import numpy as np
from sklearn.metrics.pairwise import paired_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_is_fitted

from ...ensemble import ExtraShapeletTreesClassifier, ShapeletForestClassifier
from ._nn import KNeighborsCounterfactual
from ._proto import PrototypeCounterfactual
from ._sf import ShapeletForestCounterfactual

__all__ = [
    "counterfactuals",
    "score",
    "ShapeletForestCounterfactual",
    "KNeighborsCounterfactual",
    "PrototypeCounterfactual",
]

_COUNTERFACTUALS = {
    "prototype": PrototypeCounterfactual,
}


def _best_counterfactional(estimator):
    """Infer the counterfactual explainer to use based on the estimator

    Parameters
    ----------
    estimator : object
        The estimator

    Returns
    -------
    BaseCounterfactual
        The counterfactual transformer
    """
    if isinstance(estimator, (ShapeletForestClassifier, ExtraShapeletTreesClassifier)):
        return ShapeletForestCounterfactual
    elif isinstance(estimator, KNeighborsClassifier):
        return KNeighborsCounterfactual
    else:
        return PrototypeCounterfactual


def _2d_3d_paired_distance(x, y, *, metric):
    if x.ndim != y.ndim:
        raise ValueError("both x and y must have the same rank")

    if x.ndim == 2:
        return paired_distances(x, y, metric=metric)
    elif x.ndim == 3:
        return np.mean(
            [
                paired_distances(x[:, i, :], y[:, i, :], metric=metric)
                for i in range(x.shape[1])
            ],
            axis=0,
        )
    else:
        raise ValueError("invalid rank, %d > 3" % x.ndim)


def score(x_true, x_counterfactuals, metric="euclidean", valid=None, success=None):
    """Compute the score for the counterfactuals

    Parameters
    ----------
    x_true : array-like of shape (n_samples, n_timestep)
        The true samples

    x_counterfactuals : array-like of shape (n_samples, n_timestep)
        The counterfactual samples

    metric : str, callable, list or dict, optional
        The scoring metric

        - if str use metrics from scikit-learn
        - if list compute all metrics and return a dict where the key is
          the name of the metric and the value an ndarray of scores
        - if dict compute all metrics and return a dict where the key is
          the key and the value an ndarray of scores
        - if callable

    valid : ndarray of shape (n_samples)
        Indicator matrix of successful counterfactual transformations

    Returns
    -------
    score : ndarray or dict
        The scores
    """
    # TODO: (1.2) remove success
    if success is not None:
        valid = success
        warnings.warn(
            "The success parameter has been deprecated in 1.1 "
            "and will be removed in 1.2, please use the valid parameter instead",
            FutureWarning,
        )

    if valid is not None:
        x_true = x_true[valid]
        x_counterfactuals = x_counterfactuals[valid]

    if isinstance(metric, str) or hasattr(metric, "__call__"):
        return _2d_3d_paired_distance(x_true, x_counterfactuals, metric=metric)
    else:
        sc = {}
        if isinstance(metric, dict):
            for key, value in metric.items():
                sc[key] = _2d_3d_paired_distance(
                    x_true, x_counterfactuals, metric=value
                )
        elif isinstance(metric, list):
            for item in metric:
                sc[item] = _2d_3d_paired_distance(
                    x_true, x_counterfactuals, metric=item
                )
        else:
            raise ValueError(
                "metric should be str, callable, list or dict, got %r" % metric
            )
        return sc


def counterfactuals(
    estimator,
    x,
    y,
    *,
    train_x=None,
    train_y=None,
    method="best",
    scoring=None,
    valid_scoring=False,
    random_state=None,
    method_args=None,
):
    """Compute a single counterfactual example for each sample

    Parameters
    ----------
    estimator : object
        The estimator used to compute the counterfactual example

    x : array-like of shape (n_samples, n_timestep)
        The data samples to fit counterfactuals to

    y : array-like broadcast to shape (n_samples,)
        The desired label of the counterfactual

    method : str or BaseCounterfactual, optional
        The method to generate counterfactual explanations

        - if 'best', infer the most appropriate counterfactual explanation method
          based on the estimator

          .. versionchanged :: 1.1.0

        - if str, select counterfactual explainer from named collection. See
          ``_COUNTERFACTUALS.keys()`` for a list of valid values.

        - if, BaseCounterfactual use the supplied counterfactual

    scoring : str, callable, list or dict, optional
        The scoring function to determine the similarity between the counterfactual
        sample and the original sample

    valid_scoring : bool, optional
        Only compute score for successful counterfactuals

    random_state : RandomState or int, optional
        The pseudo random number generator to ensure stable result

    method_args : dict, optional
        Optional arguments to the counterfactual explainer.

        .. versionadded :: 1.1.0

    Returns
    -------
    x_counterfactuals : ndarray of shape (n_samples, n_timestep)
        The counterfactual example.

    valid : ndarray of shape (n_samples,)
        Indicator matrix for valid counterfactuals

    score : ndarray of shape (n_samples,) or dict, optional
        Return score of the counterfactual transform, if ``scoring`` is not None
    """
    check_is_fitted(estimator)
    if method_args is None:
        method_args = {}

    # TODO: (1.2) Remove "infer"
    if isinstance(method, str):
        if method == "infer" or method == "best":
            if method == "infer":
                warnings.warn(
                    "'infer' is deprecated and should be changed "
                    "to 'best' (default). 'infer' will be disabled in 1.2.",
                    DeprecationWarning,
                )
            Explainer = _best_counterfactional(estimator)
        else:
            Explainer = _COUNTERFACTUALS.get(method)
            if Explainer is None:
                raise ValueError(
                    "method should be %s, got %r"
                    % (set(_COUNTERFACTUALS.keys()), method)
                )

        if "train_x" in method_args or "train_y" in method_args:
            train_x = method_args.pop("train_x")
            train_y = method_args.pot("train_y")
            warnings.warn(
                "train_x and train_y as method_args has been deprecated in 1.1 and "
                "will be removed in 1.2. Use the train_x and train_y keyword params.",
                FutureWarning,
            )

        explainer = Explainer(**method_args)
    else:
        explainer = method

    if random_state is not None and "random_state" in explainer.get_params():
        explainer.set_params(random_state=random_state)

    explainer.fit(estimator, train_x, train_y)
    y = np.broadcast_to(y, (x.shape[0],))
    x_counterfactuals = explainer.explain(x, y)
    valid = estimator.predict(x_counterfactuals) == y
    if scoring is not None:
        sc = score(
            x,
            x_counterfactuals,
            metric=scoring,
            valid=valid if valid_scoring else None,
        )
        return x_counterfactuals, valid, sc
    else:
        return x_counterfactuals, valid
