# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wildboar is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Authors: Isak Samsten

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.pairwise import paired_distances

from wildboar.ensemble import ShapeletForestClassifier, ExtraShapeletTreesClassifier

from ._nn import KNeighborsCounterfactual
from ._sf import ShapeletForestCounterfactual

__all__ = [
    "counterfactual",
    "score",
    "ShapeletForestCounterfactual",
    "KNeighborsCounterfactual",
]

_COUNTERFACTUAL_EXPLAINER = {}


def _infer_counterfactual(estimator):
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
        return ShapeletForestCounterfactual()
    elif isinstance(estimator, KNeighborsClassifier):
        return KNeighborsCounterfactual()
    else:
        raise NotImplemented("no support for model agnostic counterfactuals yet")


def score(x_true, counterfactuals, metric="euclidean", success=None):
    """Compute the score for the counterfactuals

    Parameters
    ----------
    x_true : array-like of shape (n_samples, n_timestep)
        The true samples

    counterfactuals : array-like of shape (n_samples, n_timestep)
        The counterfactual samples

    metric : str, callable, list or dict, optional
        The scoring metric

        - if str use metrics from scikit-learn
        - if list compute all metrics and return a dict where the key is
          the name of the metric and the value an ndarray of scores
        - if dict compute all metrics and return a dict where the key is
          the key and the value an ndarry of scores
        - if callable

    Returns
    -------
    score : ndarray or dict
        The scores
    """
    if success:
        x_true = x_true[success]
        counterfactuals = counterfactuals[success]

    if isinstance(metric, str) or hasattr(metric, "__call__"):
        return paired_distances(x_true, counterfactuals, metric=metric)
    else:
        sc = {}
        if isinstance(metric, dict):
            for key, value in metric.items():
                sc[key] = paired_distances(x_true, counterfactuals, metric=value)
        elif isinstance(metric, list):
            for item in metric:
                sc[item] = paired_distances(x_true, counterfactuals, metric=item)
        else:
            raise ValueError("invalid metric, got %r" % metric)
        return sc


def counterfactuals(
    estimator,
    x,
    y,
    *,
    method="infer",
    scoring=None,
    success_scoring=False,
    random_state=None,
    params=None
):
    """Compute a single counterfactual example for each sample

    Parameters
    ----------
    estimator : object
        The estimator used to compute the counterfactual example

    x : array-like of shape (n_samples, n_timestep) or (n_samples, n_dimension, n_timestep)
        The data samples to fit counterfactuals to

    y : array-like broadcast to shape (n_samples,)
        The desired label of the counterfactual

    method : str, optional
        The method to generate counterfactual explanations

    scoring : str, callable, list or dict, optional
        The scoring function to determine the goodness of

    success_scoring : bool, optional
        Only compute score for successful counterfactuals

    random_state : RandomState or int, optional
        The pseudo random number generator to ensure stable result

    params : dict, optional
        Optional arguments to the counterfactual explainer

    Returns
    -------
    counterfactuals : ndarray of shape (n_samples, n_timestep) or (n_samples, n_dimension, n_timestep)
        The counterfactual example.

    success : ndarray of shape (n_samples,)
        Indicator matrix for successful transformations

    score : ndarray of shape (n_samples,) or dict, optional
        Score of the counterfactual transform. Only returned if ``scoring`` is not None
    """
    check_is_fitted(estimator)
    if method == "infer":
        explainer = _infer_counterfactual(estimator)
    else:
        explainer = _COUNTERFACTUAL_EXPLAINER[method]

    if explainer is None:
        raise ValueError("no counterfactual explainer for '%r'" % method)
    y = np.broadcast_to(y, (x.shape[0],))
    explainer.set_params(random_state=random_state, **(params or {}))
    explainer.fit(estimator)
    counterfactuals, success = explainer.transform(x, y)
    if scoring:
        sc = score(
            x,
            counterfactuals,
            metric=scoring,
            success=success if success_scoring else None,
        )
        return counterfactuals, success, sc
    else:
        return counterfactuals, success
