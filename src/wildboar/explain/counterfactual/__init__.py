# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wildboar is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
# General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Authors: Isak Samsten
import warnings

import numpy as np
from sklearn.metrics.pairwise import paired_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_is_fitted

from wildboar.ensemble import ExtraShapeletTreesClassifier, ShapeletForestClassifier

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


def score(x_true, x_counterfactuals, metric="euclidean", success=None):
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

    success : ndarray of shape (n_samples)
        Indicator matrix of successful counterfactual transformations

    Returns
    -------
    score : ndarray or dict
        The scores
    """
    if success is not None:
        x_true = x_true[success]
        x_counterfactuals = x_counterfactuals[success]

    if isinstance(metric, str) or hasattr(metric, "__call__"):
        return paired_distances(x_true, x_counterfactuals, metric=metric)
    else:
        sc = {}
        if isinstance(metric, dict):
            for key, value in metric.items():
                sc[key] = paired_distances(x_true, x_counterfactuals, metric=value)
        elif isinstance(metric, list):
            for item in metric:
                sc[item] = paired_distances(x_true, x_counterfactuals, metric=item)
        else:
            raise ValueError("invalid metric, got %r" % metric)
        return sc


def counterfactuals(
    estimator,
    x,
    y,
    *,
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

    method : str, optional
        The method to generate counterfactual explanations

        - if 'best', infer the most appropriate counterfactual explanation method
          based on the estimator
          .. versionchanged :: 1.1.0
          The default parameter value is changed to 'best'

        - if str, select counterfactual explainer from named collection. See
          ``_COUNTERFACTUALS.keys()`` for a list of valid values.

    scoring : str, callable, list or dict, optional
        The scoring function to determine the similarity between the counterfactual
        sample and the original sample

    valid_scoring : bool, optional
        Only compute score for successful counterfactuals

    random_state : RandomState or int, optional
        The pseudo random number generator to ensure stable result

    method_args : dict, optional
        Optional arguments to the counterfactual explainer

        ..versionadded :: 1.1.0

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
    if method == "infer" or method == "best":
        if method == "infer":
            warnings.warn(
                "'infer' is deprecated and should be changed "
                "to 'best' (default). 'infer' will be disabled in 1.2.",
                DeprecationWarning,
            )
        Explainer = _best_counterfactional(estimator)
        if Explainer == PrototypeCounterfactual:
            warnings.warn(
                "no specific counterfactual explanation method "
                "is available for the given estimator. "
                "Using a model agnostic estimator."
            )
    else:
        Explainer = _COUNTERFACTUALS.get(method)

    if Explainer is None:
        raise ValueError("no counterfactual explainer for '%r'" % method)

    if Explainer == PrototypeCounterfactual and not (
        "train_x" in method_args or "train_y" in method_args
    ):
        raise ValueError("train_x and train_y are required in method_args")

    y = np.broadcast_to(y, (x.shape[0],))
    explainer = Explainer(**method_args)
    explainer.set_params(random_state=random_state)
    explainer.fit(estimator)
    x_counterfactuals = explainer.transform(x, y)
    success = estimator.predict(x_counterfactuals) == y
    if scoring is not None:
        sc = score(
            x,
            x_counterfactuals,
            metric=scoring,
            success=success if valid_scoring else None,
        )
        return x_counterfactuals, success, sc
    else:
        return x_counterfactuals, success
