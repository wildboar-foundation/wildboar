import abc

import numpy as np
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier as Sklearn_KNeighborsClassifier
from sklearn.utils.validation import check_is_fitted

from ...distance import KNeighborsClassifier, paired_distance
from ...ensemble import ExtraShapeletTreesClassifier, ShapeletForestClassifier
from ...utils.validation import check_array
from ._nn import KNeighborsCounterfactual
from ._proto import PrototypeCounterfactual
from ._sf import ShapeletForestCounterfactual


def make_target_evaluator(estimator, target):
    if target == "predict":
        return PredictEvaluator(estimator)
    else:
        return ProbabilityEvaluator(estimator, target)


class TargetEvaluator(abc.ABC):
    """
    Evaluate if a sample is a counterfactual.

    Parameters
    ----------
    estimator : object
        The estimator.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def is_counterfactual(self, x, y):
        """
        Return true if x is a counterfactual of label y.

        Parameters
        ----------
        x : ndarray of shape (n_timestep,)
            The counterfactual sample.
        y : object
            The counterfactual label.

        Returns
        -------
        bool
            Return true if counterfactual valid.
        """
        return self._is_counterfactual(x.reshape(1, -1), y)

    @abc.abstractmethod
    def _is_counterfactual(self, x, y):
        pass


class PredictEvaluator(TargetEvaluator):
    """Evaluate if a counterfactual is predicted as y."""

    def _is_counterfactual(self, x, y):
        return self.estimator.predict(x)[0] == y


class ProbabilityEvaluator(TargetEvaluator):
    """
    Evaluate the probability threshold.

    Parameters
    ----------
    estimator : object
        The estimator.
    probability : float, optional
        The minimum probability of the predicted label.
    """

    def __init__(self, estimator, probability=0.5):
        super().__init__(estimator)
        self.probability = probability

    def _is_counterfactual(self, x, y):
        if not hasattr(self.estimator, "predict_proba"):
            raise ValueError("estimator must support predict_proba")

        y_pred = self.estimator.predict_proba(x)
        y_idx = (self.estimator.classes_ == y).nonzero()[0][0]
        y_prob = y_pred[0, y_idx]
        return y_prob > self.probability


def _proximity(
    x_true,
    x_counterfactuals,
    metric="normalized_euclidean",
    metric_params=None,
):
    """
    Compute the proximity of the counterfactuals.

    Parameters
    ----------
    x_true : array-like of shape (n_samples, n_timestep)
        The true samples.
    x_counterfactuals : array-like of shape (n_samples, n_timestep)
        The counterfactual samples.
    metric : str or callable, optional
        The distance metric

        See ``_METRICS.keys()`` for a list of supported metrics.
    metric_params : dict, optional
        Parameters to the metric.

        Read more about the parameters in the
        :ref:`User guide <list_of_metrics>`.

    Returns
    -------
    ndarray
        The scores.
    """
    x_true = check_array(x_true, allow_3d=True, input_name="x_true")
    x_counterfactuals = check_array(
        x_counterfactuals, allow_3d=True, input_name="x_counterfactuals"
    )

    return paired_distance(
        x_true,
        x_counterfactuals,
        metric=metric,
        metric_params=metric_params,
        dim="mean",
    )


_COUNTERFACTUALS = {
    "prototype": PrototypeCounterfactual,
}


def _best_counterfactional(estimator):
    """
    Infer the counterfactual explainer to use based on the estimator.

    Parameters
    ----------
    estimator : object
        The estimator.

    Returns
    -------
    class
        The counterfactual explainer class.
    """
    if (
        isinstance(estimator, (ShapeletForestClassifier, ExtraShapeletTreesClassifier))
        and estimator.metric == "euclidean"
    ):
        return ShapeletForestCounterfactual
    elif isinstance(estimator, (Sklearn_KNeighborsClassifier, KNeighborsClassifier)):
        return KNeighborsCounterfactual
    else:
        return PrototypeCounterfactual


def counterfactuals(  # noqa: PLR0912
    estimator,
    x,
    y,
    *,
    train_x=None,
    train_y=None,
    method="best",
    proximity=None,
    random_state=None,
    method_args=None,
):
    """
    Compute a single counterfactual example for each sample.

    Parameters
    ----------
    estimator : object
        The estimator used to compute the counterfactual example.
    x : array-like of shape (n_samples, n_timestep)
        The data samples to fit counterfactuals to.
    y : array-like broadcast to shape (n_samples,)
        The desired label of the counterfactual.
    train_x : array-like of shape (n_samples, n_timestep), optional
        Training samples if required by the explainer.
    train_y : array-like of shape (n_samples, ), optional
        Training labels if required by the explainer.
    method : str or BaseCounterfactual, optional
        The method to generate counterfactual explanations

        - if 'best', infer the most appropriate counterfactual explanation
          method based on the estimator.

        .. versionchanged:: 1.1.0

        - if str, select counterfactual explainer from named collection. See
          ``_COUNTERFACTUALS.keys()`` for a list of valid values.
        - if, BaseCounterfactual use the supplied counterfactual.
    proximity : str, callable, list or dict, optional
        The scoring function to determine the similarity between the counterfactual
        sample and the original sample.
    random_state : RandomState or int, optional
        The pseudo random number generator to ensure stable result.
    method_args : dict, optional
        Optional arguments to the counterfactual explainer.

        .. versionadded:: 1.1.0

    Returns
    -------
    x_counterfactuals : ndarray of shape (n_samples, n_timestep)
        The counterfactual example.
    valid : ndarray of shape (n_samples,)
        Indicator matrix for valid counterfactuals.
    score : ndarray of shape (n_samples,) or dict, optional
        Return score of the counterfactual transform, if ``scoring`` is not None.
    """
    check_is_fitted(estimator)
    if method_args is None:
        method_args = {}

    if isinstance(method, str):
        if method == "best":
            Explainer = _best_counterfactional(estimator)
        else:
            Explainer = _COUNTERFACTUALS.get(method)
            if Explainer is None:
                raise ValueError(
                    "method should be %s, got %r"
                    % (set(_COUNTERFACTUALS.keys()), method)
                )

        explainer = Explainer(**method_args)
    else:
        explainer = clone(method)

    if random_state is not None and "random_state" in explainer.get_params():
        explainer.set_params(random_state=random_state)

    explainer.fit(estimator, train_x, train_y)
    y = np.broadcast_to(y, (x.shape[0],))
    x_counterfactuals = explainer.explain(x, y)
    valid = estimator.predict(x_counterfactuals) == y

    if proximity is not None:
        sc = _proximity(x, x_counterfactuals, metric=proximity)
        return x_counterfactuals, valid, sc
    else:
        return x_counterfactuals, valid
