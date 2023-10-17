# Authors: Isak Samsten
# License: BSD 3 clause

import abc
import math
import numbers
import warnings
from copy import deepcopy
from functools import partial

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, check_random_state

from ...base import BaseEstimator, CounterfactualMixin, ExplainerMixin
from ...distance import pairwise_subsequence_distance
from ...distance.dtw import (
    dtw_alignment,
    dtw_distance,
    dtw_mapping,
    wdtw_alignment,
    wdtw_distance,
)


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


class PrototypeSampler(abc.ABC):
    """
    Sample and refine counterfactuals.

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_timestep)
        The data samples labeled as y.
    y : object
        The label of the samples in x.
    """

    def __init__(self, x, y, prototype_indices, metric_transform):
        self.x = x
        self.y = y
        self.metric_transform = metric_transform
        self.prototype_indices = prototype_indices

    def _get_random_index(self, random_state):
        """
        Return a random index in the initial prototype sample.

        Parameters
        ----------
        random_state : RandomState
            The random state.

        Returns
        -------
        int
            The index.
        """
        return self.prototype_indices[
            random_state.randint(self.prototype_indices.shape[0])
        ]

    @abc.abstractmethod
    def sample(self, o, random_state):
        """
        Sample an example.

        Parameters
        ----------
        o : ndarray of shape (n_timestep,)
            The current counterfactual sample.
        random_state : RandomState
            The random state.

        Returns
        -------
        ndarray of shape (n_timestep,)
            A prototype of the counterfactual label.
        """
        pass

    def move(self, o, p):
        """
        Move the current counterfactual toward the prototype.

        Parameters
        ----------
        o : ndarray of shape (n_timestep,)
            The current counterfactual sample.
        p : ndarray of shape (n_timestep,)
            The prototype of the counterfactual label.

        Returns
        -------
        ndarray of shape (n_timestep,)
            The new counterfactual moved towards the prototype.
        """
        return self.metric_transform.move(o, p)

    def sample_move(self, o, random_state):
        """
        Sampla a prototype and move the counterfactual towards the prototype.

        Parameters
        ----------
        o : ndarray of shape (n_timestep,)
            The current counterfactual sample.
        random_state : RandomState
            The random state.

        Returns
        -------
        ndarray of shape (n_timestep,)
            The new counterfactual moved towards the prototype.
        """
        p = self.sample(o, random_state)
        return self.move(o, p)


class UniformPrototypeSampler(PrototypeSampler):
    """Sample a prototype uniformly at random from the initial prototype sample."""

    def sample(self, _o, random_state):
        return self.x[self._get_random_index(random_state)]


class KNearestPrototypeSampler(PrototypeSampler):
    """Sample a prototype among the samples closest to the current counterfactual."""

    def __init__(self, x, y, prototype_indicies, metric_transform):
        super().__init__(x, y, prototype_indicies, metric_transform)
        n_prototypes = prototype_indicies.size
        if isinstance(self.metric_transform, EuclideanTransform):
            self.nearest_neighbors = NearestNeighbors(
                metric="euclidean", n_neighbors=n_prototypes
            )
        elif isinstance(self.metric_transform, DynamicTimeWarpTransform):
            dtw = partial(dtw_distance, r=self.metric_transform.r)
            self.nearest_neighbors = NearestNeighbors(
                metric=dtw, n_neighbors=n_prototypes
            )
        elif isinstance(self.metric_transform, WeightedDynamicTimeWarpTransform):
            wdtw = partial(
                wdtw_distance, r=self.metric_transform.r, g=self.metric_transform.g
            )
            self.nearest_neighbors = NearestNeighbors(
                metric=wdtw, n_neighbors=n_prototypes
            )
        else:
            raise ValueError("unsupported metric")

        self.nearest_neighbors.fit(x)

    def nearest_index(self, o, random_state):
        """Return the index of the closest sample.

        Parameters
        ----------
        o : ndarray of shape (n_timestep,)
            The current counterfactual sample

        Returns
        -------
        int : an index
        """
        nearest = self.nearest_neighbors.kneighbors(
            o.reshape(1, -1), return_distance=False
        ).reshape(-1)
        return nearest[random_state.randint(nearest.shape[0])]

    def sample(self, o, random_state):
        return self.x[self.nearest_index(o, random_state)]


class ShapeletPrototypeSampler(PrototypeSampler):
    """
    Sample shapelet prototypes.

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_timestep)
        The data samples
    y : object
        The label of the samples in x
    min_shapelet_size : float
        The minimum shapelet size
    max_shapelet_size : float
            The maximum shapelet size
    """

    def __init__(
        self,
        x,
        y,
        prototype_indicies,
        metric_transform,
        min_shapelet_size=0,
        max_shapelet_size=1,
    ):
        super().__init__(x, y, prototype_indicies, metric_transform)
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size

    def sample_shapelet(self, p, random_state):
        """
        Sample a shapelet from x.

        Parameters
        ----------
        p : ndarray of shape (n_timestep,)
             The prototype sample

        Returns
        -------
        shapelet : ndarray
            A shapelet
        """
        n_timestep = self.x.shape[-1]
        min_shapelet_size = max(
            min(2, n_timestep), math.ceil(n_timestep * self.min_shapelet_size)
        )
        max_shapelet_size = math.ceil(n_timestep * self.max_shapelet_size)
        shapelet_length = random_state.randint(min_shapelet_size, max_shapelet_size)
        start_index = random_state.randint(0, n_timestep - shapelet_length)
        return p[start_index : (start_index + shapelet_length)]

    def sample(self, _o, random_state):
        return self.sample_shapelet(
            self.x[self._get_random_index(random_state)], random_state
        )

    def move(self, o, p):
        """Move the best matching shapelet towards the shapelet prototype.

        Parameters
        ----------
        o : ndarray of shape (n_timestep,)
            The counterfactual sample
        p : ndarray
            The prototype shapelet

        Returns
        -------
        new_counterfactual : ndarray of shape (n_timestep,)
            The new counterfactual moved towards the prototype
        """
        if isinstance(self.metric_transform, EuclideanTransform):
            metric = "euclidean"
            metric_params = {}
        elif isinstance(self.metric_transform, DynamicTimeWarpTransform):
            metric = "dtw"
            metric_params = {"r": self.metric_transform.r}
        elif isinstance(self.metric_transform, WeightedDynamicTimeWarpTransform):
            metric = "wdtw"
            metric_params = {"r": self.metric_transform.r, "g": self.metric_transform.g}
        else:
            raise ValueError("unsupported metric")

        # Find the best matching position in
        min_dist, best_match = pairwise_subsequence_distance(
            p,
            o,
            metric=metric,
            metric_params=metric_params,
            return_index=True,
        )
        o[best_match : best_match + p.shape[0]] = self.metric_transform.move(
            o[best_match : best_match + p.shape[0]], p
        )

        return o


class KNearestShapeletPrototypeSampler(PrototypeSampler):
    """Combines the KNearestPrototypeSample and the ShapeletPrototypeSampler.

    The prototype samples are sampled among the nearest neighbors of the
    counterfactual.
    """

    def __init__(
        self,
        x,
        y,
        prototype_indicies,
        metric_transform,
        min_shapelet_size=0,
        max_shapelet_size=1,
    ):
        super().__init__(x, y, prototype_indicies, metric_transform)
        self.nearest_sampler = KNearestPrototypeSampler(
            x, y, prototype_indicies, metric_transform
        )
        self.shapelet_sampler = ShapeletPrototypeSampler(
            x,
            y,
            prototype_indicies,
            metric_transform,
            min_shapelet_size,
            max_shapelet_size,
        )

    def sample(self, o, random_state):
        p = self.nearest_sampler.nearest_index(o, random_state)
        return self.shapelet_sampler.sample_shapelet(self.x[p], random_state)

    def move(self, o, p):
        return self.shapelet_sampler.move(o, p)


class MetricTransform(abc.ABC):
    """Move a time series towards a prototype."""

    def __init__(self, gamma):
        """Construct a new transformer.

        Parameters
        ----------
        gamma : float
            The strength of the move, where values close to 0 means that the
            sample is moved less and values closer to 1 mean that the sample
            is moved more.
        """
        self.gamma = gamma

    @abc.abstractmethod
    def move(self, o, p):
        """Move the sample o towards p.

        Parameters
        ----------
        o : ndarray of shape (n_timestep,)
            An array, the time series to move
        p : ndarray of shape (n_timestep,)
            An array, the time series to move towards

        Returns
        -------
        ndarray : an array, the result of moving o closer to p
        """
        pass


class EuclideanTransform(MetricTransform):
    """Transform a sample by moving it closer in euclidean space."""

    def move(self, o, p):
        return ((1 - self.gamma) * o) + self.gamma * p


class DynamicTimeWarpTransform(MetricTransform):
    """Transform a sample by moving it closer using the dtw alignment matrix."""

    def __init__(self, gamma, r=1.0):
        super().__init__(gamma)
        self.r = r

    def move(self, o, p):
        _, indices = dtw_mapping(alignment=self._get_alignment(o, p), return_index=True)

        # This a fast but somewhat crude approximation of DBA (as implemented) in
        # dtw_average with sample_weight=[(1 - gamma), gamma] and init=0 but
        # significantly faster to compute.
        result = o * (1 - self.gamma)
        weight = np.ones(o.shape[0]) * (1 - self.gamma)
        for i, j in zip(indices[0], indices[1]):
            result[i] += self.gamma * p[j]
            weight[i] += self.gamma

        return result / weight

    def _get_alignment(self, o, p):
        return dtw_alignment(o, p, r=self.r)


class WeightedDynamicTimeWarpTransform(DynamicTimeWarpTransform):
    def __init__(self, gamma, r=1, g=0.05):
        super().__init__(gamma, r)
        self.g = g

    def _get_alignment(self, o, p):
        return wdtw_alignment(o, p, r=self.r, g=self.g)


_METRIC_TRANSFORM = {
    "euclidean": EuclideanTransform,
    "dtw": DynamicTimeWarpTransform,
    "wdtw": WeightedDynamicTimeWarpTransform,
}

_PROTOTYPE_SAMPLER = {
    "sample": UniformPrototypeSampler,
    "nearest": KNearestPrototypeSampler,
    "shapelet": ShapeletPrototypeSampler,
    "nearest_shapelet": KNearestShapeletPrototypeSampler,
}


class PrototypeCounterfactual(CounterfactualMixin, ExplainerMixin, BaseEstimator):
    """Model agnostic approach for constructing counterfactual explanations.

    Attributes
    ----------
    estimator_ : object
        The estimator for which counterfactuals are computed
    classes_ : ndarray
        The classes
    partitions_ : dict
        Dictionary of classes and PrototypeSampler
    target_ : TargetEvaluator
        The target evaluator

    References
    ----------
    Samsten, Isak (2020).
        Model agnostic time series counterfactuals
    """

    _parameter_constraints: dict = {
        "metric": [StrOptions(_METRIC_TRANSFORM.keys())],
        "r": [None, Interval(numbers.Real, 0, 1.0, closed="both")],
        "g": [None, Interval(numbers.Real, 0, None, closed="right")],
        "max_iter": [Interval(numbers.Integral, 1, None, closed="left")],
        "step_size": [Interval(numbers.Real, 0, 1, closed="right")],
        "n_prototypes": [
            StrOptions({"auto"}),
            Interval(numbers.Integral, 1, None, closed="left"),
            Interval(numbers.Real, 0, 1, closed="right"),
        ],
        "target": [
            StrOptions({"predict", "auto"}, deprecated={"auto"}),
            Interval(numbers.Real, 0.5, 1, closed="right"),
        ],
        "method": [StrOptions(_PROTOTYPE_SAMPLER.keys())],
        "min_shapelet_size": [Interval(numbers.Real, 0, 1, closed="left")],
        "max_shapelet_size": [Interval(numbers.Real, 0, 1, closed="right")],
        "random_state": ["random_state"],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        metric="euclidean",
        *,
        r=1.0,
        g=0.05,
        max_iter=100,
        step_size=0.1,
        n_prototypes="auto",
        target="predict",
        method="sample",
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        random_state=None,
        verbose=False,
    ):
        """Crate a new model agnostic counterfactual explainer.

        Parameters
        ----------
        metric : {'euclidean', 'dtw', 'wdtw'}, optional
            The metric used to move the samples
        r : float, optional
            The warping window size, if metric='dtw' or metric='wdtw'
        g : float, optional
            Penalty control for weighted DTW, if metric='wdtw'
        max_iter : int, optional
            The maximum number of iterations
        step_size : float, optional
            The step size when moving samples toward class prototypes
        n_prototypes : int, float or str, optional
            The number of initial prototypes to sample from
        target : float or {'predict'}, optional
            The target evaluation of counterfactuals:

            - if 'predict' the counterfactual prediction must return the correct
              label
            - if float, the counterfactual prediction probability must
              exceed target value
        method : {'sample', 'shapelet', 'nearest', 'nearest_shapelet'}, optional
            Method for selecting prototypes

            - if 'sample' a prototype is sampled among the initial prototypes
            - if 'shapelet' a prototype shapelet is sampled among the initial
              prototypes
            - if 'nearest' a prototype is sampled from the closest n prototypes
            - if 'nearest_shapelet' a prototype shapelet is sampled from the
              closest n prototypes
        min_shapelet_size : float, optional
            Minimum shapelet size, if method='shapelet' or 'nearest_shapelet'
        max_shapelet_size : float, optional
            Maximum shapelet size, if method='shapelet' or 'nearest_shapelet'
        random_state : RandomState or int, optional
            Pseudo-random number for consistency between different runs
        """
        self.random_state = random_state
        self.metric = metric
        self.r = r
        self.g = g
        self.max_iter = max_iter
        self.step_size = step_size
        self.n_prototypes = n_prototypes
        self.method = method
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.target = target
        self.verbose = verbose

    def fit(self, estimator, x, y):
        if x is None or y is None:
            raise ValueError("Both training samples and labels are required.")

        self._validate_params()
        estimator = self._validate_estimator(estimator)
        x, y = self._validate_data(x, y, reset=False, dtype=float)

        metric_params = {}
        if self.metric in ["dtw", "wdtw"]:
            metric_params["r"] = self.r
        if self.metric == ["wdtw"]:
            metric_params["g"] = self.g

        random_state = check_random_state(self.random_state)
        metric_transform = _METRIC_TRANSFORM[self.metric](
            self.step_size,
            **metric_params,
        )
        Sampler = _PROTOTYPE_SAMPLER[self.method]

        self.estimator_ = deepcopy(estimator)
        self.classes_ = np.unique(y)
        self.random_state_ = random_state.randint(np.iinfo(np.int32).max)
        if self.target in {"auto", "predict"}:
            if self.target == "auto":
                warnings.warn(
                    f"The parameter value 'auto' for target of {type(self).__name__} "
                    "has been renamed to 'predict' in 1.2 and will be removed in 1.4.",
                    DeprecationWarning,
                )
            self.target_ = PredictEvaluator(self.estimator_)
        else:
            self.target_ = ProbabilityEvaluator(self.estimator_, self.target)

        if self.method in ["shapelet", "nearest_shapelet"]:
            if self.min_shapelet_size > self.max_shapelet_size:
                raise ValueError(
                    f"The parameter min_shapelet_size of {type(self).__name__} must be "
                    "<= max_shapelet_size."
                )

            method_params = {
                "min_shapelet_size": self.min_shapelet_size,
                "max_shapelet_size": self.max_shapelet_size,
            }
        else:
            method_params = {}

        self.partitions_ = {}
        for c in self.classes_:
            x_partition = x[y == c]
            if self.n_prototypes == "auto":
                n_prototypes = x_partition.shape[0]
            elif isinstance(self.n_prototypes, numbers.Integral):
                n_prototypes = max(1, min(self.n_prototypes, x_partition.shape[0]))
            else:
                n_prototypes = math.ceil(self.n_prototypes * x_partition.shape[0])

            prototype_indicies = np.arange(x_partition.shape[0])
            random_state.shuffle(prototype_indicies)
            prototype_indicies = prototype_indicies[:n_prototypes]
            self.partitions_[c] = Sampler(
                x_partition,
                c,
                prototype_indicies,
                metric_transform,
                **method_params,
            )

    def explain(self, x, y):
        check_is_fitted(self)
        random_state = check_random_state(self.random_state_)
        x, y = self._validate_data(x, y, reset=False, dtype=float)
        counterfactuals = np.empty(x.shape, dtype=x.dtype)
        for i in range(x.shape[0]):
            if self.verbose:
                print(f"Computing counterfactual for the {i}th sample.")

            counterfactuals[i] = self._transform_sample(
                x[i], y[i], random_state.randint(np.iinfo(np.int32).max)
            )

        return counterfactuals

    def _transform_sample(self, x, y, random_state):
        random_state = check_random_state(random_state)
        sampler = self.partitions_[y]
        o = x.copy()
        n_iter = 0
        while not self.target_.is_counterfactual(o, y) and n_iter < self.max_iter:
            if self.verbose and n_iter % (self.max_iter // 10) == 0:
                print(f"Running {n_iter}/{self.max_iter}...")

            o = sampler.sample_move(o, random_state)
            n_iter += 1

        if self.verbose:
            print(f"Completed after {n_iter} iterations.")
            if n_iter == self.max_iter:
                print(
                    "The counterfactual explain reached max_iter, increase step_size "
                    "or max_iter for convergence."
                )
        return o

    def _more_tags():
        return {"requires_y": True}
