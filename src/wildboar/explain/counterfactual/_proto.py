# Authors: Isak Samsten
# License: BSD 3 clause

import abc
import math
import numbers
import warnings
from copy import deepcopy

import numpy as np
from scipy.special import softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, check_random_state

from ...base import BaseEstimator, CounterfactualMixin, ExplainerMixin
from ...distance import (
    NearestNeighbors,
    pairwise_distance,
    pairwise_subsequence_distance,
)
from ...distance.dtw import (
    dtw_alignment,
    dtw_mapping,
    wdtw_alignment,
)
from ...transform import ShapeletTransform


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
        if hasattr(self.estimator, "classes_"):
            self.threshold = 1.0 / len(self.estimator.classes_)
        else:
            self.threshold = 0.5

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

    def evaluate(self, x, y):
        return self.is_counterfactual(x, y), self.get_score(x, y)

    def get_score(self, x, y):
        """
        Get the score (probability or confidence) for the target class.

        Parameters
        ----------
        x : ndarray of shape (n_timestep,)
            The counterfactual sample.
        y : object
            The counterfactual label.

        Returns
        -------
        float or None
            Score for the target class, or None if scoring is not supported.
        """
        return self._get_score(x.reshape(1, -1), y)

    @abc.abstractmethod
    def _is_counterfactual(self, x, y):
        pass

    @abc.abstractmethod
    def _get_score(self, x, y):
        pass


class PredictEvaluator(TargetEvaluator):
    """Evaluate if a counterfactual is predicted as y."""

    def _is_counterfactual(self, x, y):
        return self.estimator.predict(x)[0] == y

    def _get_score(self, x, y):
        """Return None for hard predictions as there is no continuous score."""
        if hasattr(self.estimator, "predict_proba"):
            y_pred = self.estimator.predict_proba(x)
            y_idx = (self.estimator.classes_ == y).nonzero()[0][0]
            return y_pred[0, y_idx]
        return None


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
        self.threshold = probability

    def _is_counterfactual(self, x, y):
        if not hasattr(self.estimator, "predict_proba"):
            raise ValueError("estimator must support predict_proba")

        y_pred = self.estimator.predict_proba(x)
        y_idx = (self.estimator.classes_ == y).nonzero()[0][0]
        y_prob = y_pred[0, y_idx]
        return y_prob > self.probability

    def _get_score(self, x, y):
        """Return the probability of the target class."""
        if not hasattr(self.estimator, "predict_proba"):
            raise ValueError("estimator must support predict_proba")

        y_pred = self.estimator.predict_proba(x)
        y_idx = (self.estimator.classes_ == y).nonzero()[0][0]
        return y_pred[0, y_idx]

    def evaluate(self, x, y):
        score = self.get_score(x, y)
        return score > self.probability, score


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

    def __init__(self, x, y, prototype_indices, *, metric, metric_params):
        self.x = x
        self.y = y
        self.prototype_indices = prototype_indices
        self.metric = metric
        self.metric_params = metric_params if metric_params is not None else {}
        self.move_factory = _METRIC_TRANSFORM[self.metric]

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
        idx = random_state.randint(self.prototype_indices.shape[0])
        return self.prototype_indices[idx]

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

    def sample_transform(self, o, random_state):
        """
        Sample a prototype and return a callable that can apply the move with different gammas.

        Parameters
        ----------
        o : ndarray of shape (n_timestep,)
            The current counterfactual sample.
        random_state : RandomState
            The random state.

        Returns
        -------
        callable
            A function that takes a gamma parameter and returns the transformed sample.
        """
        p = self.sample(o, random_state)
        return self.move_factory(p, **self.metric_params)


class UniformPrototypeSampler(PrototypeSampler):
    """Sample a prototype uniformly at random from the initial prototype sample."""

    def sample(self, _o, random_state):
        return self.x[self._get_random_index(random_state)]


class KNearestPrototypeSampler(PrototypeSampler):
    """Sample a prototype among the samples closest to the current counterfactual."""

    def __init__(self, x, y, prototype_indices, *, metric, metric_params, temperature):
        super().__init__(
            x, y, prototype_indices, metric=metric, metric_params=metric_params
        )
        self.temperature = temperature

        if metric == "scaled_euclidean":
            metric = "euclidean"
        self.nearest_neighbors = NearestNeighbors(
            metric=metric,
            metric_params=metric_params,
            n_neighbors=prototype_indices.size,
        )
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
        distances, indices = self.nearest_neighbors.kneighbors(
            o.reshape(1, -1), return_distance=True
        )
        distances = distances.reshape(-1)
        indices = indices.reshape(-1)
        p = softmax(-distances / self.temperature)
        return random_state.choice(indices, p=p)

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
        prototype_indices,
        *,
        metric,
        metric_params,
        min_shapelet_size=0,
        max_shapelet_size=1,
    ):
        super().__init__(
            x, y, prototype_indices, metric=metric, metric_params=metric_params
        )
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

    def sample_transform(self, o, random_state):
        p = self.sample(o, random_state)
        return ShapeletMetricTransform(
            self.metric, self.metric_params, self.move_factory(p, **self.metric_params)
        )


class DiscriminativeShapeletSampler(ShapeletPrototypeSampler):
    """
    Sample discriminative shapelet prototypes.

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_timestep)
        The data samples
    y : object
        The label of the samples in x
    prototype_indices : ndarray
        The indices of the prototypes
    metric_transform : MetricTransform
        The metric transform
    shapelets : list of ndarray
        The discriminative shapelets
    """

    def __init__(
        self,
        x,
        y,
        prototype_indices,
        *,
        metric,
        metric_params,
        shapelets,
        min_shapelet_size=0,
        max_shapelet_size=1,
    ):
        super().__init__(
            x,
            y,
            prototype_indices,
            metric=metric,
            metric_params=metric_params,
            min_shapelet_size=min_shapelet_size,
            max_shapelet_size=max_shapelet_size,
        )
        self.shapelets = shapelets

    def sample(self, _o, random_state):
        return self.shapelets[random_state.randint(len(self.shapelets))]


class KNearestShapeletPrototypeSampler(ShapeletPrototypeSampler):
    """Combines the KNearestPrototypeSample and the ShapeletPrototypeSampler.

    The prototype samples are sampled among the nearest neighbors of the
    counterfactual.
    """

    def __init__(
        self,
        x,
        y,
        prototype_indices,
        *,
        metric,
        metric_params,
        temperature,
        min_shapelet_size=0,
        max_shapelet_size=1,
    ):
        super().__init__(
            x, y, prototype_indices, metric=metric, metric_params=metric_params
        )
        self.nearest_sampler = KNearestPrototypeSampler(
            x,
            y,
            prototype_indices,
            metric=metric,
            metric_params=metric_params,
            temperature=temperature,
        )

    def sample(self, o, random_state):
        idx = self.nearest_sampler.nearest_index(o, random_state)
        return self.sample_shapelet(self.x[idx], random_state)


class MetricTransform(abc.ABC):
    """Move a time series towards a prototype."""

    def __init__(self, prototype):
        self.prototype = prototype

    @abc.abstractmethod
    def move(self, o, step):
        """Move the sample o towards prototype.

        Parameters
        ----------
        o : ndarray of shape (n_timestep,)
            An array, the time series to move
        step : float
            The distance to move toward the prototype. The implementation
            converts this to an interpolation coefficient gamma based on
            the distance between o and the prototype.

        Returns
        -------
        ndarray : an array, the result of moving o closer to p
        """
        pass

    @abc.abstractmethod
    def reverse_move(self, o, original, gamma):
        """Move the sample o back towards the original.

        This is used during post-processing to reduce distance to the original
        while maintaining counterfactual validity.

        Parameters
        ----------
        o : ndarray of shape (n_timestep,)
            The current state (typically a counterfactual).
        original : ndarray of shape (n_timestep,)
            The original sample we want to move back toward.
        gamma : float
            The strength of the reverse move, where values close to 0 means
            minimal movement and values closer to 1 mean larger steps back.

        Returns
        -------
        ndarray : an array, the result of moving o closer to original
        """
        pass


class ShapeletMetricTransform:
    def __init__(self, metric, metric_params, metric_transform):
        self.metric = metric
        self.metric_params = metric_params
        self.metric_transform = metric_transform

    def _get_shapelet_region(self, o):
        """Find where the shapelet best matches in o."""
        prototype = self.metric_transform.prototype
        _, start = pairwise_subsequence_distance(
            prototype,
            o,
            metric=self.metric,
            metric_params=self.metric_params,
            return_index=True,
        )
        end = start + prototype.shape[0]
        return start, end

    def move(self, o, step):
        new_o = o.copy()
        start, end = self._get_shapelet_region(o)
        new_o[start:end] = self.metric_transform.move(o[start:end], step)
        return new_o

    def reverse_move(self, o, original, gamma):
        """Move the shapelet region back toward the original's corresponding region."""
        new_o = o.copy()
        start, end = self._get_shapelet_region(o)
        # Move this region back toward the original's corresponding region
        new_o[start:end] = self.metric_transform.reverse_move(
            o[start:end], original[start:end], gamma
        )
        return new_o


class EuclideanTransform(MetricTransform):
    """Transform a sample by moving it closer in euclidean space."""

    def move(self, o, step):
        dist_to_proto = np.linalg.norm(self.prototype - o)
        if dist_to_proto < 1e-10:
            return o.copy()
        gamma = min(1.0, step / dist_to_proto)
        return ((1 - gamma) * o) + gamma * self.prototype

    def reverse_move(self, o, original, gamma):
        return ((1 - gamma) * o) + gamma * original


class ScaledEuclideanTransform(MetricTransform):
    """Transform a sample by moving it closer in z-normalized euclidean space."""

    def __init__(self, prototype):
        super().__init__(prototype)
        self.prototype_std = np.std(prototype)
        if self.prototype_std < 1e-13:
            self.prototype_std = 1.0
        self.prototype_mean = np.mean(prototype)
        self.prototype_norm = (prototype - self.prototype_mean) / self.prototype_std

    def move(self, o, step):
        std_o = np.std(o)
        if std_o < 1e-13:
            std_o = 1.0
        mean_o = np.mean(o)
        o_norm = (o - mean_o) / std_o

        # Compute distance in normalized space to determine gamma
        dist_to_proto = np.linalg.norm(self.prototype_norm - o_norm)
        if dist_to_proto < 1e-10:
            return o.copy()
        gamma = min(1.0, step / dist_to_proto)

        new_norm = ((1 - gamma) * o_norm) + gamma * self.prototype_norm
        new_std = ((1 - gamma) * std_o) + gamma * self.prototype_std
        new_mean = ((1 - gamma) * mean_o) + gamma * self.prototype_mean

        return new_norm * new_std + new_mean

    def reverse_move(self, o, original, gamma):
        # Compute stats for original
        std_orig = np.std(original)
        if std_orig < 1e-13:
            std_orig = 1.0
        mean_orig = np.mean(original)
        orig_norm = (original - mean_orig) / std_orig

        # Compute stats for current state
        std_o = np.std(o)
        if std_o < 1e-13:
            std_o = 1.0
        mean_o = np.mean(o)
        o_norm = (o - mean_o) / std_o

        # Interpolate in normalized space back toward original
        new_norm = ((1 - gamma) * o_norm) + gamma * orig_norm
        new_std = ((1 - gamma) * std_o) + gamma * std_orig
        new_mean = ((1 - gamma) * mean_o) + gamma * mean_orig

        return new_norm * new_std + new_mean


class DynamicTimeWarpTransform(MetricTransform):
    """Transform a sample by moving it closer using the dtw alignment matrix."""

    def __init__(self, prototype, r=1.0):
        super().__init__(prototype)
        self.r = r

    def move(self, o, step):
        # Get the DTW alignment path between original sample (o) and prototype (p)
        _, (o_indices, p_indices) = dtw_mapping(
            alignment=self._get_alignment(o), return_index=True
        )

        # Count how many prototype timesteps align to each sample timestep
        count = np.bincount(o_indices, minlength=o.shape[0])
        aligned_sum = np.bincount(
            o_indices, weights=self.prototype[p_indices], minlength=o.shape[0]
        )

        # Compute the average of aligned prototype values
        aligned_avg = np.zeros(o.shape[0])
        mask = count > 0
        aligned_avg[mask] = aligned_sum[mask] / count[mask]

        # Compute distance to aligned target to determine gamma
        dist_to_aligned = np.linalg.norm(aligned_avg - o)
        if dist_to_aligned < 1e-10:
            return o.copy()
        gamma = min(1.0, step / dist_to_aligned)

        # Apply the weighted combination
        result = (1 - gamma) * o + gamma * aligned_avg

        return result

    def reverse_move(self, o, original, gamma):
        """Move back toward original using DTW alignment."""
        # Get the DTW alignment path between current state (o) and original
        alignment = dtw_alignment(o, original, r=self.r)
        _, (o_indices, orig_indices) = dtw_mapping(
            alignment=alignment, return_index=True
        )

        # Count how many original timesteps align to each current timestep
        count = np.bincount(o_indices, minlength=o.shape[0])
        aligned_sum = np.bincount(
            o_indices, weights=original[orig_indices], minlength=o.shape[0]
        )

        # Compute the average of aligned original values
        aligned_avg = np.zeros(o.shape[0])
        mask = count > 0
        aligned_avg[mask] = aligned_sum[mask] / count[mask]

        # Apply the weighted combination back toward original
        result = (1 - gamma) * o + gamma * aligned_avg

        return result

    def _get_alignment(self, o):
        return dtw_alignment(o, self.prototype, r=self.r)


class WeightedDynamicTimeWarpTransform(DynamicTimeWarpTransform):
    def __init__(self, prototype, r=1.0, g=0.05):
        super().__init__(prototype, r=r)
        self.g = g

    def _get_alignment(self, o):
        return wdtw_alignment(o, self.prototype, r=self.r, g=self.g)

    def reverse_move(self, o, original, gamma):
        """Move back toward original using weighted DTW alignment."""
        alignment = wdtw_alignment(o, original, r=self.r, g=self.g)
        _, (o_indices, orig_indices) = dtw_mapping(
            alignment=alignment, return_index=True
        )

        count = np.bincount(o_indices, minlength=o.shape[0])
        aligned_sum = np.bincount(
            o_indices, weights=original[orig_indices], minlength=o.shape[0]
        )

        aligned_avg = np.zeros(o.shape[0])
        mask = count > 0
        aligned_avg[mask] = aligned_sum[mask] / count[mask]

        return (1 - gamma) * o + gamma * aligned_avg


_METRIC_TRANSFORM = {
    "euclidean": EuclideanTransform,
    "scaled_euclidean": ScaledEuclideanTransform,
    "dtw": DynamicTimeWarpTransform,
    "wdtw": WeightedDynamicTimeWarpTransform,
}

_PROTOTYPE_SAMPLER = {
    "sample": UniformPrototypeSampler,
    "nearest": KNearestPrototypeSampler,
    "shapelet": ShapeletPrototypeSampler,
    "nearest_shapelet": KNearestShapeletPrototypeSampler,
    "discriminative_shapelet": DiscriminativeShapeletSampler,
}


class CostFunction:
    """
    Unified counterfactual cost function with optional adaptive Lipschitz estimation.

    The cost function J(s) combines two terms:

    - g(s): Normalized distance from original = ||s - x|| / ||x||
    - h(s): Normalized margin to decision boundary

    This can be interpreted in two ways:

    1. **Weighted sum**: J = w*g + (1-w)*h
       where w = distance_weight

    2. **Lipschitz A***: J = g + h (when w=0.5 and default L)
       The heuristic h estimates minimum distance to decision boundary
       based on Lipschitz continuity of the probability function.

    Parameters
    ----------
    x_original : ndarray
        Original sample being explained.
    threshold : float
        Target counterfactual class.
    distance_weight : float, default=0.5
        Weight for distance term. Higher values prioritize staying close
        to the original; lower values prioritize crossing the boundary.
    adaptive : bool, default=False
        If True, adaptively estimate Lipschitz constant from observed gradients.
    lipschitz : float, optional
        Initial Lipschitz estimate. If None (default), uses threshold/||x||
        which matches the non-adaptive assumption.
    lipschitz_smoothing : float, default=0.1
        EMA smoothing factor for Lipschitz updates (only used if adaptive=True).

    Notes
    -----
    When adaptive=False (default), h(s) = margin / threshold, which
    assumes the probability changes by `threshold` over distance ||x||.
    This is equivalent to Lipschitz constant L = threshold / ||x||.

    When adaptive=True, the Lipschitz constant is estimated from observed
    probability changes during search, and h(s) = (margin / L) / ||x||,
    representing the estimated minimum distance to reach the boundary.
    """

    def __init__(
        self,
        x_original,
        *,
        threshold,
        distance_weight=0.5,
        adaptive=False,
        lipschitz=None,
        lipschitz_smoothing=0.1,
        mean_distance=None,
    ):
        self.threshold = threshold
        self.x_original = x_original
        self.distance_weight = distance_weight
        self.adaptive = adaptive

        self.x_norm = np.linalg.norm(x_original)
        if self.x_norm == 0:
            self.x_norm = 1.0

        if mean_distance is not None and mean_distance > 0:
            self.min_step_size = 0.02 * mean_distance
            self.max_step_size = 0.5 * mean_distance
        else:
            self.min_step_size = 0.01 * self.x_norm
            self.max_step_size = 0.3 * self.x_norm

        if lipschitz is None:
            scale = mean_distance if mean_distance is not None else self.x_norm
            self.lipschitz = threshold / max(scale, 1e-8)
        else:
            self.lipschitz = lipschitz
        self.lipschitz_smoothing = lipschitz_smoothing

    def get_distance_estimate(self, prob):
        """
        Estimate the distance needed to reach the decision boundary.

        Uses the Lipschitz assumption to estimate: d_need = margin / L.

        Parameters
        ----------
        prob : float or None
            Current probability for target class.

        Returns
        -------
        float
            Estimated distance to boundary.
        """
        if prob is None:
            return 0.1 * (self.max_step_size + self.min_step_size)

        prob = np.clip(prob, 0.0, 1.0)
        margin = max(0.0, self.threshold - prob)

        if margin <= 1e-9:
            return self.min_step_size

        L = np.clip(self.lipschitz, 1e-9, np.inf)
        d_need = margin / L

        return np.clip(d_need, self.min_step_size, self.max_step_size)

    def evaluate(self, state, prob, is_cf):
        """
        Compute cost for a state.

        Parameters
        ----------
        state : ndarray
            Current sample state.
        prob : float, optional
            Pre-computed probability for the state. If None, will be computed
            using the target evaluator.

        Returns
        -------
        float
            Cost in [0, 1]. Lower is better.
        """
        distance = np.linalg.norm(state - self.x_original)
        g = distance / self.x_norm

        if is_cf:
            h = 0.0
        elif prob is None:
            h = 1.0
        else:
            prob = float(np.clip(prob, 0.0, 1.0))
            margin = max(0.0, self.threshold - prob)
            L = np.clip(self.lipschitz, 1e-8, np.inf)
            h = (margin / L) / self.x_norm

        return self.distance_weight * g + (1.0 - self.distance_weight) * h

    def update(self, state_old, prob_old, state_new, prob_new):
        """
        Update Lipschitz estimate from observed transition.

        Parameters
        ----------
        state_old : ndarray
            Previous state.
        prob_old : float or None
            Probability at previous state.
        state_new : ndarray
            New state.
        prob_new : float or None
            Probability at new state.
        """
        if not self.adaptive:
            return

        if prob_old is None or prob_new is None:
            return

        delta_prob = abs(prob_new - prob_old)
        delta_dist = np.linalg.norm(state_new - state_old)

        if delta_dist <= 1e-10 or delta_prob <= 1e-10:
            return

        local_L = delta_prob / delta_dist
        self.lipschitz = np.clip(
            (1 - self.lipschitz_smoothing) * self.lipschitz
            + self.lipschitz_smoothing * local_L,
            1e-8,
            np.inf,
        )


_REFINERS = {}


def _register_refiner(name):
    """Decorator to register a built-in refiner."""

    def decorator(func):
        _REFINERS[name] = func
        return func

    return decorator


@_register_refiner("path")
def path_refine(cf, x_original, y_target, path, target_evaluator, max_iter):
    """
    Refine counterfactual by reversing along the path of transforms.

    Attempts to partially undo each transform in the path (in reverse order),
    accepting changes that reduce distance while maintaining validity.

    Parameters
    ----------
    cf : ndarray
        The counterfactual to refine.
    x_original : ndarray
        The original sample.
    y_target : int
        The target class.
    path : list of MetricTransform
        The sequence of transforms applied to reach cf.
    target_evaluator : TargetEvaluator
        Evaluator to check counterfactual validity.
    max_iter : int
        Maximum iterations (used as multiplier for path length).

    Returns
    -------
    ndarray
        Refined counterfactual.
    """
    if not target_evaluator.is_counterfactual(cf, y_target):
        return cf

    current = cf.copy()
    best_distance = np.linalg.norm(current - x_original)

    for _ in range(max_iter):
        improved = False
        for transform in reversed(path):
            for gamma in [0.1, 0.2, 0.3, 0.4, 0.5]:
                candidate = transform.reverse_move(current, x_original, gamma)

                if target_evaluator.is_counterfactual(candidate, y_target):
                    dist = np.linalg.norm(candidate - x_original)
                    if dist < best_distance:
                        current = candidate
                        best_distance = dist
                        improved = True
                        break
            if improved:
                break

        if not improved:
            break

    return current


@_register_refiner("global")
def global_refine(cf, x_original, y_target, path, target_evaluator, max_iter):
    """
    Refine counterfactual by nudging toward original.

    Iteratively moves the counterfactual closer to the original sample
    using small interpolation steps, accepting changes that reduce distance
    while maintaining validity.

    Parameters
    ----------
    cf : ndarray
        The counterfactual to refine.
    x_original : ndarray
        The original sample.
    y_target : int
        The target class.
    path : list of MetricTransform
        The sequence of transforms (unused, for consistent interface).
    target_evaluator : TargetEvaluator
        Evaluator to check counterfactual validity.
    max_iter : int
        Maximum refinement iterations.

    Returns
    -------
    ndarray
        Refined counterfactual.
    """
    if not target_evaluator.is_counterfactual(cf, y_target):
        return cf

    current = cf.copy()
    best_distance = np.linalg.norm(current - x_original)

    for _ in range(max_iter):
        improved = False
        for gamma in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
            # Direct interpolation toward original
            candidate = (1 - gamma) * current + gamma * x_original

            if target_evaluator.is_counterfactual(candidate, y_target):
                dist = np.linalg.norm(candidate - x_original)
                if dist < best_distance:
                    current = candidate
                    best_distance = dist
                    improved = True
                    break
            else:
                break

        if not improved:
            break

    return current


def apply_refinement(
    cf, x_original, y_target, path, target_evaluator, refine, refine_iter
):
    """
    Apply refinement strategy to a counterfactual.

    Parameters
    ----------
    cf : ndarray
        The counterfactual to refine.
    x_original : ndarray
        The original sample.
    y_target : int
        The target class.
    path : list of MetricTransform
        The sequence of transforms applied to reach cf.
    target_evaluator : TargetEvaluator
        Evaluator to check counterfactual validity.
    refine : str, callable, list, or None
        Refinement strategy:
        - None: no refinement
        - str: built-in refiner name ("path", "global")
        - callable: custom refiner function
        - list: chain of refiners applied in order
    refine_iter : int
        Number of iterations for built-in refiners.

    Returns
    -------
    ndarray
        Refined counterfactual.
    """
    if refine is None:
        return cf

    if isinstance(refine, str):
        refiners = [refine]
    elif callable(refine):
        refiners = [refine]
    else:
        refiners = list(refine)

    current = cf
    for r in refiners:
        if isinstance(r, str):
            if r not in _REFINERS:
                raise ValueError(
                    f"Unknown refiner '{r}'. Available: {list(_REFINERS.keys())}"
                )
            current = _REFINERS[r](
                current, x_original, y_target, path, target_evaluator, refine_iter
            )
        elif callable(r):
            current = r(current, x_original, y_target, path, target_evaluator)
        else:
            raise TypeError(f"Refiner must be str or callable, got {type(r).__name__}")

    return current


class OptimizationStrategy(abc.ABC):
    """
    Abstract base class for counterfactual search optimization strategies.

    Parameters
    ----------
    target_evaluator : TargetEvaluator
        Evaluator to check if counterfactual and get scores.
    step_size : float
        Default step size for moves.
    """

    def __init__(self, target_evaluator, step_size, mean_distance=None):
        self.target_evaluator = target_evaluator
        self.step_size = step_size
        self.mean_distance = mean_distance

    @abc.abstractmethod
    def optimize(self, x, y, sampler, random_state):
        """
        Search for a counterfactual.

        Parameters
        ----------
        x : ndarray of shape (n_timestep,)
            Original sample to explain.
        y : int
            Target counterfactual class.
        sampler : PrototypeSampler
            Sampler that provides move closures via sample_transform(o, random_state).
        random_state : RandomState
            Random state for reproducibility.

        Returns
        -------
        result : ndarray of shape (n_timestep,)
            The counterfactual sample, or the original if no CF found.
        path : list of MetricTransform
            The sequence of transforms applied to reach the result.
        """
        pass


class GreedySearch(OptimizationStrategy):
    """
    Simple greedy search with backtracking line search.

    This strategy samples a move direction at each iteration and uses
    line search to find an optimal step size. It accepts moves that
    improve the score toward the target class.

    Parameters
    ----------
    target_evaluator : TargetEvaluator
        Evaluator to check if counterfactual and get scores.
    step_size : float
        Initial step size (distance) for moves.
    max_iter : int
        Maximum number of search iterations.
    line_search_iter : int, optional
        Maximum number of line search iterations per move. Default is 10.
    step_increase : float, optional
        Factor to increase step when move succeeds. Default is 1.2.
    step_decrease : float, optional
        Factor to decrease step when move fails. Default is 0.5.
    min_step : float, optional
        Minimum step to try before giving up on a direction. Default is 1e-4.
    """

    def __init__(
        self,
        target_evaluator,
        step_size,
        max_iter,
        mean_distance=None,
        line_search_iter=10,
        step_increase=1.2,
        step_decrease=0.5,
        min_step=1e-4,
    ):
        super().__init__(target_evaluator, step_size, mean_distance=mean_distance)
        self.max_iter = max_iter
        self.line_search_iter = line_search_iter
        self.step_increase = step_increase
        self.step_decrease = step_decrease
        self.min_step = min_step

    def optimize(self, x, y, sampler, random_state):
        x_original = x.copy()
        o = x.copy()
        path = []
        cost = CostFunction(
            x_original,
            threshold=self.target_evaluator.threshold,
            adaptive=True,
            mean_distance=self.mean_distance,
        )

        for _ in range(self.max_iter):
            is_cf, prob = self.target_evaluator.evaluate(o, y)
            transform = sampler.sample_transform(o, random_state)
            o_new, crossed, used_transform = self._line_search(
                o, transform, y, cost, prob
            )
            if used_transform:
                path.append(transform)
            if crossed:
                return o_new, path

            o = o_new

        return o, path

    def _line_search(self, current, transform, y, cost, current_prob):
        current_cost = cost.evaluate(
            current, current_prob, self.target_evaluator.is_counterfactual(current, y)
        )
        best = current.copy()
        best_cost = current_cost
        crossed = False
        improved = False

        if self.step_size == "adaptive":
            step = cost.get_distance_estimate(current_prob)
        else:
            step = self.step_size

        for _ in range(self.line_search_iter):
            if step < self.min_step:
                break

            candidate = transform.move(current, step)
            is_cf, prob = self.target_evaluator.evaluate(candidate, y)

            cost.update(current, current_prob, candidate, prob)

            if is_cf:
                crossed = True
                best = candidate
                improved = True
                break

            new_cost = cost.evaluate(candidate, prob, is_cf)
            if new_cost < best_cost:
                best = candidate
                best_cost = new_cost
                step *= self.step_increase
                improved = True
            else:
                step *= self.step_decrease

        return best, crossed, improved


class _BeamState:
    __slots__ = ("state", "path", "score")

    def __init__(self, state, path, score):
        self.state = state
        self.path = path
        self.score = score


class BeamSearch(OptimizationStrategy):
    """
    Beam search for counterfactual generation.

    Maintains a beam of top-k candidates and expands all of them at each
    iteration. This provides diversity in exploration while focusing on
    promising states.

    Parameters
    ----------
    target_evaluator : TargetEvaluator
        Evaluator to check if counterfactual and get scores.
    step_size : float
        Step size (distance) for moves.
    max_iter : int
        Maximum number of search iterations.
    beam_width : int, optional
        Number of candidates to keep at each iteration. Default is 5.
    n_branches : int, optional
        Number of moves to try from each state during expansion. Default is 3.
    """

    def __init__(
        self,
        target_evaluator,
        step_size,
        max_iter,
        mean_distance=None,
        beam_width=5,
        n_branches=3,
    ):
        super().__init__(target_evaluator, step_size, mean_distance=mean_distance)
        self.max_iter = max_iter
        self.beam_width = beam_width
        self.n_branches = n_branches

    def optimize(self, x, y, sampler, random_state):
        """
        Perform beam search for counterfactual.

        At each iteration:
        1. Expand all states in beam with multiple moves
        2. Score all candidates
        3. Keep top beam_width candidates
        4. Return if counterfactual found

        Parameters
        ----------
        x : ndarray of shape (n_timestep,)
            Original sample.
        y : int
            Target class.
        sampler : PrototypeSampler
            Sampler for moves.
        random_state : RandomState
            Random state.

        Returns
        -------
        result : ndarray of shape (n_timestep,)
            Best counterfactual found, or best candidate if none found.
        path : list of MetricTransform
            The sequence of transforms applied.
        """
        cost = CostFunction(
            x,
            threshold=self.target_evaluator.threshold,
            adaptive=True,
            mean_distance=self.mean_distance,
        )
        is_cf, prob = self.target_evaluator.evaluate(x, y)
        initial_score = self._score(x, cost, prob, is_cf)
        beam = [_BeamState(x.copy(), [], initial_score)]

        best_cf = None
        best_cf_distance = float("inf")
        best_cf_path = []

        for _ in range(self.max_iter):
            candidates = []

            for beam_state in beam:
                if self.target_evaluator.is_counterfactual(beam_state.state, y):
                    dist = np.linalg.norm(beam_state.state - x)
                    if dist < best_cf_distance:
                        best_cf = beam_state.state.copy()
                        best_cf_distance = dist
                        best_cf_path = beam_state.path.copy()

                current_prob = self.target_evaluator.get_score(beam_state.state, y)
                for _ in range(self.n_branches):
                    transform = sampler.sample_transform(beam_state.state, random_state)
                    if self.step_size == "adaptive":
                        step = cost.get_distance_estimate(current_prob)
                    else:
                        step = self.step_size
                    new_state = transform.move(beam_state.state, step)
                    new_path = beam_state.path + [transform]

                    is_cf, prob = self.target_evaluator.evaluate(new_state, y)
                    cost.update(beam_state.state, current_prob, new_state, prob)

                    new_score = self._score(new_state, cost, prob, is_cf)
                    candidates.append(_BeamState(new_state, new_path, new_score))

            if not candidates:
                break

            candidates.sort(key=lambda s: s.score, reverse=True)
            beam = candidates[: self.beam_width]

        for beam_state in beam:
            if self.target_evaluator.is_counterfactual(beam_state.state, y):
                dist = np.linalg.norm(beam_state.state - x)
                if dist < best_cf_distance:
                    best_cf = beam_state.state.copy()
                    best_cf_distance = dist
                    best_cf_path = beam_state.path.copy()

        if best_cf is not None:
            return best_cf, best_cf_path

        if beam:
            return beam[0].state, beam[0].path
        return x, []

    def _score(self, state, cost, prob, is_cf):
        """
        Parameters
        ----------
        state : ndarray
            State to score.
        cost : CostFunction
            Cost function for evaluation.
        prob : float, optional
            Pre-computed probability for the state.

        Returns
        -------
        float
            Score (higher is better). CFs score in (1, 2], non-CFs in [0, 1].
        """
        return 1.0 - cost.evaluate(state, prob, is_cf)


class BestFirstSearch(OptimizationStrategy):
    """
    Best-first search for counterfactual generation.

    Uses A* style scoring with adaptive Lipschitz estimation to prioritize
    which states to expand. The cost function combines distance from the
    original sample with an estimate of the minimum distance needed to
    reach the decision boundary.

    Parameters
    ----------
    target_evaluator : TargetEvaluator
        Evaluator for checking counterfactual validity.
    step_size : float
        Step size for moves.
    max_iter : int, optional
        Maximum number of states to expand. Default is 100.
    n_branches : int, optional
        Number of moves to try per expansion. Default is 10.
    proximity_weight : float, optional
        Weight for distance term in cost function. Default is 0.5.
        Higher values prioritize staying close to original.
    lipschitz_init : float, optional
        Initial Lipschitz constant estimate. If None (default), uses
        threshold/||x|| which matches the non-adaptive assumption.
    lipschitz_momentum : float, optional
        Momentum factor for Lipschitz EMA updates. Default is 0.1.
    """

    def __init__(
        self,
        target_evaluator,
        step_size,
        max_iter=100,
        mean_distance=None,
        n_branches=10,
        proximity_weight=0.5,
        lipschitz_init=None,
        lipschitz_momentum=0.1,
    ):
        super().__init__(target_evaluator, step_size, mean_distance=mean_distance)
        self.max_iter = max_iter
        self.n_branches = n_branches
        self.proximity_weight = proximity_weight
        self.lipschitz_init = lipschitz_init
        self.lipschitz_momentum = lipschitz_momentum

    def optimize(self, x, y, sampler, random_state):
        """
        Perform best-first search for a counterfactual.

        Uses adaptive Lipschitz estimation to improve the A* heuristic
        as the search progresses.

        Parameters
        ----------
        x : ndarray
            Original sample.
        y : int
            Target class.
        sampler : PrototypeSampler
            Sampler for generating moves.
        random_state : RandomState
            Random state for reproducibility.

        Returns
        -------
        result : ndarray
            Best counterfactual found, or original if none found.
        path : list of MetricTransform
            The sequence of transforms applied.
        """
        import heapq

        x_original = x.copy()

        cost = CostFunction(
            x_original,
            threshold=self.target_evaluator.threshold,
            distance_weight=self.proximity_weight,
            adaptive=True,
            lipschitz=self.lipschitz_init,
            lipschitz_smoothing=self.lipschitz_momentum,
            mean_distance=self.mean_distance,
        )

        counter = 0
        is_cf, initial_prob = self.target_evaluator.evaluate(x, y)
        initial_cost = cost.evaluate(x, initial_prob, is_cf)
        # (cost, counter, state, path, prob)
        heap = [(initial_cost, counter, x.copy(), [], initial_prob)]
        counter += 1

        visited = set()
        visited.add(x.tobytes())

        best_cf = None
        best_cf_path = []
        best_cf_distance = float("inf")

        expansions = 0
        while heap and expansions < self.max_iter:
            _prev_cost, _, state, path, current_prob = heapq.heappop(heap)

            if self.target_evaluator.is_counterfactual(state, y):
                dist = np.linalg.norm(state - x_original)
                if dist < best_cf_distance:
                    best_cf = state.copy()
                    best_cf_path = path.copy()
                    best_cf_distance = dist
                continue

            expansions += 1
            for _ in range(self.n_branches):
                if self.step_size == "adaptive":
                    step = cost.get_distance_estimate(current_prob)
                else:
                    step = self.step_size
                transform = sampler.sample_transform(state, random_state)
                new_state = transform.move(state, step)

                state_key = new_state.tobytes()
                if state_key in visited:
                    continue
                visited.add(state_key)

                is_cf, new_prob = self.target_evaluator.evaluate(new_state, y)

                cost.update(state, current_prob, new_state, new_prob)

                new_cost = cost.evaluate(new_state, new_prob, is_cf)
                new_path = path + [transform]
                heapq.heappush(heap, (new_cost, counter, new_state, new_path, new_prob))
                counter += 1

        if best_cf is not None:
            return best_cf, best_cf_path

        if heap:
            _, _, best_state, best_path, _ = heapq.heappop(heap)
            return best_state, best_path

        return x_original, []


class PrototypeCounterfactual(CounterfactualMixin, ExplainerMixin, BaseEstimator):
    """Model agnostic approach for constructing counterfactual explanations.

    This method generates counterfactual explanations by iteratively moving
    a sample toward prototypes of the target class until the classifier's
    prediction changes. The movement is guided by the selected metric
    (Euclidean or DTW-based), and the process can be optimized using
    different search strategies.

    Parameters
    ----------
    metric : {'euclidean', 'dtw', 'wdtw'}, optional
        The distance metric used to compute the movement direction toward
        prototypes.

        - 'euclidean': Standard Euclidean distance. Fast but sensitive to
          temporal misalignment.
        - 'dtw': Dynamic Time Warping. Handles temporal shifts but slower.
        - 'wdtw': Weighted DTW with penalty for large warping. Balances
          flexibility and stability.

    r : float, optional
        The Sakoe-Chiba warping window size as a fraction of the time series
        length. Only used when ``metric='dtw'`` or ``metric='wdtw'``. A value
        of 1.0 allows unrestricted warping, while smaller values constrain
        the warping path to stay close to the diagonal. Default is 1.0.

    g : float, optional
        The penalty parameter for weighted DTW. Only used when
        ``metric='wdtw'``. Controls how much large warpings are penalized.
        Larger values result in behavior closer to Euclidean distance.
        Default is 0.05.

    max_iter : int, optional
        Maximum number of iterations for the optimization process. The search
        terminates when this limit is reached, even if no valid counterfactual
        has been found. Default is 100.

    step_size : {"adaptive"} or float, optional
        Controls how far the sample moves toward the prototype at each step.

        - 'adaptive': Automatically estimates an appropriate step size using
          a Lipschitz-based estimate of the distance to the decision boundary.
          This typically leads to faster convergence.
        - float: Uses a fixed step size computed as ``step_size * mean_dist``,
          where ``mean_dist`` is the mean positive pairwise distance in the
          training data. For example, ``step_size=0.2`` moves 20% of the
          typical inter-sample spacing per iteration.

        Default is 'adaptive'.

    n_prototypes : int, float, or {'auto'}, optional
        The number of prototypes to consider when selecting movement targets.

        - 'auto': Automatically determines the number based on dataset size.
        - int: Uses exactly this many prototypes per class.
        - float: Uses this fraction of training samples per class as prototypes.

        Default is 'auto'.

    sampling_temperature : float, optional
        Temperature parameter for prototype sampling when ``method`` is
        'nearest' or 'nearest_shapelet'. Controls the randomness of prototype
        selection using softmax weighting: ``exp(-distance / temperature)``.

        - High values (e.g., 10.0): More uniform sampling across prototypes.
        - Low values (e.g., 0.1): Greedy selection of nearest prototypes.
        - 1.0: Balanced sampling weighted by distance.

        Default is 1.0.

    target : float or {'predict'}, optional
        Defines when a counterfactual is considered successful.

        - 'predict': The counterfactual is valid when the classifier's
          predicted label changes to the target class.
        - float (0.5-1.0): The counterfactual is valid when the predicted
          probability for the target class exceeds this threshold. Higher
          values produce more confident counterfactuals but may require
          larger modifications.

        Default is 'predict'.

    method : {'sample', 'shapelet', 'nearest', 'nearest_shapelet', 'discriminative_shapelet'}, optional
        Strategy for selecting prototypes to move toward.

        - 'sample': Uniformly samples a prototype from all training samples
          of the target class.
        - 'shapelet': Samples a shapelet (subsequence) from a randomly
          selected prototype. Produces more localized changes.
        - 'nearest': Samples from the k-nearest prototypes to the current
          sample, weighted by distance and temperature.
        - 'nearest_shapelet': Combines 'nearest' and 'shapelet' - samples
          shapelets from nearby prototypes.
        - 'discriminative_shapelet': Uses shapelets that are most
          discriminative between classes, based on Cohen's d effect size.

        Default is 'sample'.

    optimizer : {'greedy', 'best_first', 'beam'}, optional
        Search strategy for finding counterfactuals.

        - 'greedy': At each step, commits to a single move. Fast but may
          get stuck in local optima.
        - 'best_first': Explores the most promising states first using a
          priority queue. More thorough but slower.
        - 'beam': Maintains multiple candidate solutions in parallel,
          keeping the top ``beam_width`` candidates at each step. Balances
          exploration and efficiency.

        Default is 'greedy'.

    beam_width : int, optional
        Number of candidate solutions to maintain when ``optimizer='beam'``.
        Larger values explore more of the search space but increase
        computation time. Default is 5.

    n_branches : int, optional
        Number of different moves to try from each state during expansion.
        Only used when ``optimizer`` is 'best_first' or 'beam'. Higher values
        increase diversity but slow down the search. Default is 3.

    refine : str, list, callable, or None, optional
        Refinement strategy applied after finding a valid counterfactual to
        minimize the distance from the original sample while maintaining
        validity.

        - None: No refinement. Returns the counterfactual as-is.
        - 'path': Backtracks along the search path to find the minimal
          modification that still produces a valid counterfactual.
        - 'global': Nudges the counterfactual back toward the original sample
          in small steps until validity is lost.
        - callable: Custom refinement function with signature
          ``(cf, x_original, y_target, path, target_evaluator) -> cf_refined``.
        - list: Applies multiple refinements in sequence (e.g.,
          ``['path', 'global']``).

        Default is ``['path', 'global']``.

    refine_iter : int, optional
        Maximum number of iterations for each built-in refinement strategy.
        Higher values may produce closer counterfactuals but increase
        computation time. Ignored for custom callable refinements.
        Default is 5.

    min_shapelet_size : float, optional
        Minimum shapelet length as a fraction of the time series length.
        Only used when ``method`` is 'shapelet', 'nearest_shapelet', or
        'discriminative_shapelet'. Default is 0.0 (no minimum).

    max_shapelet_size : float, optional
        Maximum shapelet length as a fraction of the time series length.
        Only used when ``method`` is 'shapelet', 'nearest_shapelet', or
        'discriminative_shapelet'. Default is 1.0 (full length).

    n_shapelets : int, optional
        Number of candidate shapelets to extract during shapelet discovery.
        Only used when ``method='discriminative_shapelet'``. More candidates
        increase the chance of finding discriminative patterns but slow down
        fitting. Default is 1000.

    k_best_shapelets : int, optional
        Number of top-scoring shapelets to retain per class. Only used when
        ``method='discriminative_shapelet'``. Shapelets are ranked by Cohen's
        d effect size. Default is 10.

    random_state : int, RandomState instance, or None, optional
        Controls randomness for reproducibility. Pass an int for reproducible
        results across multiple function calls. Default is None.

    n_jobs : int or None, optional
        Number of parallel jobs for shapelet extraction. Only used when
        ``method='discriminative_shapelet'``. None means 1 job, -1 means
        using all processors. Default is None.

    Attributes
    ----------
    estimator_ : object
        The fitted estimator for which counterfactuals are computed.

    classes_ : ndarray of shape (n_classes,)
        The unique class labels from the training data.

    partitions_ : dict
        Dictionary mapping each class label to its corresponding
        PrototypeSampler instance containing the prototypes for that class.

    target_ : TargetEvaluator
        The evaluator used to determine if a candidate counterfactual is
        valid (either PredictEvaluator or ProbabilityEvaluator depending
        on the ``target`` parameter).

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
        "step_size": [
            StrOptions({"adaptive"}),
            Interval(numbers.Real, 0, None, closed="neither"),
        ],
        "n_prototypes": [
            StrOptions({"auto"}),
            Interval(numbers.Integral, 1, None, closed="left"),
            Interval(numbers.Real, 0, 1, closed="right"),
        ],
        "target": [
            StrOptions({"predict"}),
            Interval(numbers.Real, 0.5, 1, closed="right"),
        ],
        "method": [StrOptions(_PROTOTYPE_SAMPLER.keys())],
        "sampling_temperature": [Interval(numbers.Real, 0, None, closed="neither")],
        "optimizer": [StrOptions({"greedy", "beam", "best_first"})],
        "beam_width": [Interval(numbers.Integral, 1, None, closed="left")],
        "n_branches": [Interval(numbers.Integral, 1, None, closed="left")],
        "refine": [None, str, list, callable],
        "refine_iter": [Interval(numbers.Integral, 1, None, closed="left")],
        "min_shapelet_size": [Interval(numbers.Real, 0, 1, closed="left")],
        "max_shapelet_size": [Interval(numbers.Real, 0, 1, closed="right")],
        "n_shapelets": [Interval(numbers.Integral, 1, None, closed="left")],
        "k_best_shapelets": [Interval(numbers.Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
        "n_jobs": [None, numbers.Integral],
    }

    def __init__(
        self,
        metric="euclidean",
        *,
        r=1.0,
        g=0.05,
        max_iter=100,
        step_size="adaptive",
        n_prototypes="auto",
        target="predict",
        method="sample",
        sampling_temperature=1.0,
        optimizer="greedy",
        beam_width=5,
        n_branches=3,
        refine=["path", "global"],
        refine_iter=5,
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        n_shapelets=1000,
        k_best_shapelets=10,
        random_state=None,
        n_jobs=None,
    ):
        self.random_state = random_state
        self.metric = metric
        self.r = r
        self.g = g
        self.max_iter = max_iter
        self.step_size = step_size
        self.n_prototypes = n_prototypes
        self.method = method
        self.sampling_temperature = sampling_temperature
        self.optimizer = optimizer
        self.beam_width = beam_width
        self.n_branches = n_branches
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.n_shapelets = n_shapelets
        self.k_best_shapelets = k_best_shapelets
        self.target = target
        self.n_jobs = n_jobs
        self.refine = refine
        self.refine_iter = refine_iter

    def fit(self, estimator, x, y):
        if x is None or y is None:
            raise ValueError("Both training samples and labels are required.")

        self._validate_params()
        estimator = self._validate_estimator(estimator)
        x, y = self._validate_data(x, y, reset=False, dtype=float)

        metric_params = {}
        if self.metric in ["dtw", "wdtw"]:
            metric_params["r"] = self.r
        if self.metric == "wdtw":
            metric_params["g"] = self.g

        random_state = check_random_state(self.random_state)
        Sampler = _PROTOTYPE_SAMPLER[self.method]

        self.estimator_ = deepcopy(estimator)
        self.classes_ = np.unique(y)
        self.random_state_ = random_state.randint(np.iinfo(np.int32).max)
        if isinstance(self.target, str) and self.target == "predict":
            self.target_ = PredictEvaluator(self.estimator_)
        else:
            self.target_ = ProbabilityEvaluator(self.estimator_, self.target)

        if self.method in ["shapelet", "nearest_shapelet"]:
            if self.min_shapelet_size > self.max_shapelet_size:
                raise ValueError(
                    f"The parameter min_shapelet_size of {type(self).__name__} must be "
                    "<= max_shapelet_size."
                )

        if self.method == "discriminative_shapelet":
            shapelet_transform = ShapeletTransform(
                metric=self.metric,
                metric_params=metric_params,
                n_shapelets=self.n_shapelets,
                min_shapelet_size=self.min_shapelet_size,
                max_shapelet_size=self.max_shapelet_size,
                n_jobs=self.n_jobs,
                random_state=random_state.randint(np.iinfo(np.int32).max),
            ).fit(x, y)

            X_dist = shapelet_transform.transform(x)
            attributes = shapelet_transform.embedding_.attributes
            shapelets = {}
            for c in self.classes_:
                scores = []
                for i in range(X_dist.shape[1]):
                    dist_c = X_dist[y == c, i]
                    dist_not_c = X_dist[y != c, i]
                    if len(dist_c) > 0 and len(dist_not_c) > 0:
                        # Cohen's d: effect size measure
                        mean_diff = np.mean(dist_not_c) - np.mean(dist_c)
                        pooled_std = np.sqrt((np.var(dist_c) + np.var(dist_not_c)) / 2)
                        score = mean_diff / pooled_std if pooled_std > 0 else 0
                    else:
                        score = -np.inf
                    scores.append(score)

                scores = np.array(scores)
                best_indices = np.argsort(scores)[-self.k_best_shapelets :]
                shapelets[c] = [attributes[i][1][1] for i in best_indices]

        pw = pairwise_distance(x, metric=self.metric)
        pw_pos = pw[pw > 0]
        self.mean_distance_ = float(np.mean(pw_pos)) if len(pw_pos) > 0 else 1.0

        self.partitions_ = {}
        for c in self.classes_:
            x_partition = x[y == c]
            if self.n_prototypes == "auto":
                n_prototypes = x_partition.shape[0]
            elif isinstance(self.n_prototypes, numbers.Integral):
                n_prototypes = max(1, min(self.n_prototypes, x_partition.shape[0]))
            else:
                n_prototypes = math.ceil(self.n_prototypes * x_partition.shape[0])

            prototype_indices = np.arange(x_partition.shape[0])
            random_state.shuffle(prototype_indices)
            prototype_indices = prototype_indices[:n_prototypes]

            method_params = {}
            if self.method in {"shapelet", "nearest_shapelet"}:
                method_params = {
                    "min_shapelet_size": self.min_shapelet_size,
                    "max_shapelet_size": self.max_shapelet_size,
                }
            if self.method == "discriminative_shapelet":
                method_params["shapelets"] = shapelets[c]

            if self.method in {"nearest", "nearest_shapelet"}:
                method_params["temperature"] = self.sampling_temperature

            self.partitions_[c] = Sampler(
                x_partition,
                c,
                prototype_indices,
                metric=self.metric,
                metric_params=metric_params,
                **method_params,
            )

        if isinstance(self.step_size, str) and self.step_size == "adaptive":
            resolved_step_size = "adaptive"
        else:
            resolved_step_size = float(self.step_size) * self.mean_distance_

        if self.optimizer == "greedy":
            self.optimizer_ = GreedySearch(
                max_iter=self.max_iter,
                target_evaluator=self.target_,
                step_size=resolved_step_size,
                mean_distance=self.mean_distance_,
            )
        elif self.optimizer == "beam":
            self.optimizer_ = BeamSearch(
                target_evaluator=self.target_,
                max_iter=self.max_iter,
                step_size=resolved_step_size,
                mean_distance=self.mean_distance_,
                beam_width=self.beam_width,
                n_branches=self.n_branches,
            )
        elif self.optimizer == "best_first":
            self.optimizer_ = BestFirstSearch(
                target_evaluator=self.target_,
                step_size=resolved_step_size,
                max_iter=self.max_iter,
                mean_distance=self.mean_distance_,
                n_branches=self.n_branches,
            )
        return self

    def explain(self, x, y):
        check_is_fitted(self)
        random_state = check_random_state(self.random_state_)
        x, y = self._validate_data(x, y, reset=False, dtype=float)
        counterfactuals = np.empty(x.shape, dtype=x.dtype)
        for i in range(x.shape[0]):
            counterfactuals[i] = self._transform_sample(
                x[i], y[i], random_state.randint(np.iinfo(np.int32).max)
            )

        return counterfactuals

    def _transform_sample(self, x, y, random_state):
        random_state = check_random_state(random_state)

        sampler = self.partitions_[y]
        cf, path = self.optimizer_.optimize(x, y, sampler, random_state)

        cf = apply_refinement(
            cf, x, y, path, self.target_, self.refine, self.refine_iter
        )

        if not self.target_.is_counterfactual(cf, y):
            warnings.warn(
                "The counterfactual explain reached max_iter without finding a "
                "counterfactual. Increase step_size or max_iter for convergence.",
                ConvergenceWarning,
            )

        return cf

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        return tags
