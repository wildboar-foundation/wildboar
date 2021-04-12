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
import abc
import math
from copy import deepcopy
from functools import partial

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state

from wildboar.distance import distance
from wildboar.distance.dtw import dtw_distance, dtw_mapping

from .base import BaseCounterfactual


class PrototypeCounterfactual(BaseCounterfactual):
    """Model agnostic approach for constructing counterfactual explanations

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

    def __init__(
        self,
        background_x,
        background_y,
        *,
        metric="euclidean",
        metric_params=None,
        max_iter=100,
        step_size=0.1,
        n_prototypes="auto",
        target="auto",
        method="sample",
        method_params=None,
        random_state=None
    ):
        """Crate a new model agnostic counterfactual explainer.

        Parameters
        ----------
        background_x : array-like of shape (n_samples, n_timestep)
            The background data from which prototypes are sampled

        background_y : array-like of shape (n_samples,)
            The background label from which prototypes are sampled

        metric : {'euclidean', 'dtw'}, optional
            The metric used to move the samples

        metric_params : dict, optional
            Optional parameters to the metric

            If 'dtw':

                r : int or float, optional
                    The warping window size

        max_iter : int, optional
            The maximum number of iterations

        step_size : float, optional
            The step size when moving samples toward class prototypes

        n_prototypes : int, float or str, optional
            The number of initial prototypes to sample from

        target : float or str, optional
            The target evaluation of counterfactuals:

            - if 'auto' the counterfactual prediction must return the correct
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

        method_params : dict, optional
            Additional parameters to the method

            If 'shapelet' or 'nearest_shapelet'

                min_shapelet_size : float, optional
                    Minimum shapelet size.

                max_shapelet_size : float, optional
                    Maximum shapelet size.

        random_state : RandomState or int, optional
            Pseudo-random number for consistency between different runs
        """
        self.background_x = background_x
        self.background_y = background_y
        self.random_state = random_state
        self.metric = metric
        self.metric_params = metric_params
        self.max_iter = max_iter
        self.step_size = step_size
        self.n_prototypes = n_prototypes
        self.method = method
        self.method_params = method_params
        self.target = target

    def fit(self, estimator):
        check_is_fitted(estimator)
        if self.background_x is None or self.background_y is None:
            raise ValueError("background data are required.")

        x = check_array(self.background_x)
        y = check_array(self.background_y, ensure_2d=False)
        if len(y) != x.shape[0]:
            raise ValueError(
                "Number of labels={} does not match "
                "number of samples={}".format(len(y), x.shape[0])
            )
        random_state = check_random_state(self.random_state)
        metric_params = self.metric_params or {}
        method_params = self.method_params or {}
        if self.metric in _METRIC_TRANSFORM:
            metric = _METRIC_TRANSFORM[self.metric](self.step_size, **metric_params)
        else:
            raise ValueError("metric (%s) is not supported" % self.metric)

        if self.method in _PROTOTYPE_SAMPLER:
            sampler = _PROTOTYPE_SAMPLER[self.method]
        else:
            raise ValueError("method (%s) is not supported" % self.method)

        self.estimator_ = deepcopy(estimator)
        self.classes_ = np.unique(self.background_y)
        if self.target == "auto":
            self.target_ = PredictEvaluator(self.estimator_)
        else:
            if not 0 < self.target <= 1.0:
                raise ValueError("target must be in (0, 1], got %r" % self.target)
            self.target_ = ProbabilityEvaluator(self.estimator_, self.target)

        self.partitions_ = {}
        for c in self.classes_:
            x_partition = x[y == c]
            if self.n_prototypes == "auto":
                n_prototypes = x_partition.shape[0]
            elif isinstance(self.n_prototypes, int):
                n_prototypes = max(1, min(self.n_prototypes, x_partition.shape[0]))
            elif isinstance(self.n_prototypes, float):
                if not 0.0 < self.n_prototypes <= 1.0:
                    raise ValueError("n_prototypes")
                n_prototypes = math.ceil(self.n_prototypes * x_partition.shape[0])
            else:
                raise ValueError("n_prototypes (%r) not supported" % self.n_prototypes)

            self.partitions_[c] = sampler(
                x_partition, c, n_prototypes, metric, random_state, **method_params
            )

    def transform(self, x, y):
        x = check_array(x)
        y = check_array(y, ensure_2d=False)
        if len(y) != x.shape[0]:
            raise ValueError(
                "Number of labels={} does not match "
                "number of samples={}".format(len(y), x.shape[0])
            )
        n_samples = x.shape[0]
        counterfactuals = np.empty(x.shape, dtype=x.dtype)
        success = np.ones(x.shape[0]).astype(bool)
        for i in range(n_samples):
            counterfactual = self._transform_sample(x[i], y[i])
            if counterfactual is not None:
                counterfactuals[i] = counterfactual
            else:
                success[i] = False
        return counterfactuals, success

    def _transform_sample(self, x, y):
        sampler = self.partitions_[y]
        o = x.copy()
        n_iter = 0
        while not self.target_.is_counterfactual(o, y) and n_iter < self.max_iter:
            o = sampler.sample_move(o)
            n_iter += 1

        if n_iter > self.max_iter:
            return None
        else:
            return o


class TargetEvaluator(abc.ABC):
    """Evaluate if a sample is a counterfactual"""

    def __init__(self, estimator):
        """Construct a new evaluator

        Parameters
        ----------
        estimator : object
            The estimator
        """
        self.estimator = estimator

    def is_counterfactual(self, x, y):
        """Return true if x is a counterfactual of label y

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_timestep) or (n_timestep,)
            The counterfactual sample

        y : object
            The counterfactual label

        Returns
        -------
        bool : true if counterfactual
        """
        return self._is_counterfactual(x.reshape(1, -1), y)

    @abc.abstractmethod
    def _is_counterfactual(self, x, y):
        pass


class PredictEvaluator(TargetEvaluator):
    """Evaluate if a counterfactual is predicted as y"""

    def _is_counterfactual(self, x, y):
        return self.estimator.predict(x) == y


class ProbabilityEvaluator(TargetEvaluator):
    """Evaluate if the probability of a counterfactual is at least a given constant"""

    def __init__(self, estimator, probability=0.5):
        """Construct a new evaluator

        Parameters
        ----------
        estimator : object
            The estimator

        probability : float
            The minimum probability of the predicted label
        """
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
    def __init__(self, x, y, n_prototypes, metric_transform, random_state):
        """Sample and refine counterfactuals

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_timestep)
            The data samples labeled as y

        y : object
            The label of the samples in x

        n_prototypes : int
            The number of prototypes in the initial sample

        metric_transform : MetricTransform
            The metric transformer.

        random_state : RandomState
            The random number generator.
        """
        self.random_state = random_state
        self.x = x
        self.y = y
        self.metric_transform = metric_transform
        self.n_prototypes = n_prototypes
        prototype_indices = np.arange(x.shape[0])
        self.random_state.shuffle(prototype_indices)
        self.prototype_indices = prototype_indices[:n_prototypes]

    @property
    def _random_index(self):
        """Return a random index in the initial prototype sample

        Returns
        -------
        int : an index
        """
        return self.prototype_indices[
            self.random_state.randint(self.prototype_indices.shape[0])
        ]

    def _random_indices(self, n):
        """Return n random indices from the initial prototype sample

        Parameters
        ----------
        n : int
            The number of idices to return

        Returns
        -------
        ndarray : indices
        """
        n = min(self.n_prototypes - 1, max(1, n))
        self.random_state.shuffle(self.prototype_indices)
        return self.prototype_indices[:n]

    @abc.abstractmethod
    def sample(self, o):
        """Sample an example

        Parameters
        ----------
        o : ndarray of shape (n_timestep,)
            The current counterfactual sample

        Returns
        -------
        prototype : ndarray of shape (n_timestep,)
            A prototype of the counterfactual label
        """
        pass

    def move(self, o, p):
        """Move the current counterfactual toward the prototype

        Parameters
        ----------
        o : ndarray of shape (n_timestep,)
            The current counterfactual sample

        p : ndarray of shape (n_timestep,)
            The prototype of the counterfactual label

        Returns
        -------
        new_counterfactual : ndarray of shape (n_timestep,)
            The new counterfactual moved towards the prototype
        """
        return self.metric_transform.move(o, p)

    def sample_move(self, o):
        """Sampla a prototype and move the counterfactual towards the prototype

        Parameters
        ----------
        o : ndarray of shape (n_timestep,)
            The current counterfactual sample

        Returns
        -------
        new_counterfactual : ndarray of shape (n_timestep,)
            The new counterfactual moved towards the prototype
        """
        p = self.sample(o)
        return self.move(o, p)


class UniformPrototypeSampler(PrototypeSampler):
    """Sample a prototype uniformly at random from the initial prototype sample"""

    def sample(self, _o):
        return self.x[self._random_index]


class KNearestPrototypeSampler(PrototypeSampler):
    """Sample a prototype among the samples closest to the current counterfactual"""

    def __init__(self, x, y, n_prototypes, metric_transform, random_state):
        super().__init__(x, y, n_prototypes, metric_transform, random_state)
        if isinstance(self.metric_transform, EuclideanTransform):
            self.nearest_neighbors = NearestNeighbors(
                metric="euclidean", n_neighbors=n_prototypes
            )
        elif isinstance(self.metric_transform, DynamicTimeWarpTransform):
            dtw = partial(dtw_distance, r=self.metric_transform.r)
            self.nearest_neighbors = NearestNeighbors(
                metric=dtw, n_neighbors=n_prototypes
            )
        else:
            raise ValueError("unsupported metric")

        self.nearest_neighbors.fit(x)

    def nearest_index(self, o):
        """Return the index of the closest sample

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
        return nearest[self.random_state.randint(nearest.shape[0])]

    def sample(self, o):
        return self.x[self.nearest_index(o)]


class ShapeletPrototypeSampler(PrototypeSampler):
    """Sample shapelet prototypes"""

    def __init__(
        self,
        x,
        y,
        n_prototypes,
        metric_transform,
        random_state,
        min_shapelet_size=0,
        max_shapelet_size=1,
    ):
        """Sample shapelet

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_timestep)
            The data samples

        y : object
            The label of the samples in x

        metric_transform : MetricTransform
            The metric transformer.

        random_state : RandomState
            The random number generator.

        min_shapelet_size : float
            The minimum shapelet size

        max_shapelet_size : float
            The maximum shapelet size
        """
        super().__init__(x, y, n_prototypes, metric_transform, random_state)
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size

    def sample_shapelet(self, p):
        """Sample a shapelet from x

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
        min_shapelet_size = max(2, int(n_timestep * self.min_shapelet_size))
        max_shapelet_size = int(n_timestep * self.max_shapelet_size)
        shapelet_length = self.random_state.randint(
            min_shapelet_size, max_shapelet_size
        )
        start_index = self.random_state.randint(0, n_timestep - shapelet_length)
        return p[start_index : (start_index + shapelet_length)]

    def sample(self, _o):
        return self.sample_shapelet(self.x[self._random_index])

    def move(self, o, p):
        """Move the best matching shapelet of the  counterfactual sample towards
        the shapelet prototype

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
        else:
            raise ValueError("unsupported metric")

        # Find the best matching position in
        min_dist, best_match = distance(
            p, o, metric=metric, metric_params=metric_params, return_index=True
        )
        o[best_match : best_match + p.shape[0]] = self.metric_transform.move(
            o[best_match : best_match + p.shape[0]], p
        )

        return o


class KNearestShapeletPrototypeSampler(PrototypeSampler):
    """Combines the KNearestPrototypeSample and the ShapeletPrototypeSampler
    such that prototype samples are sampled among the nearest neighbors of the
    counterfactual
    """

    def __init__(
        self,
        x,
        y,
        n_prototypes,
        metric_transform,
        random_state,
        min_shapelet_size=0,
        max_shapelet_size=1,
    ):
        super().__init__(x, y, n_prototypes, metric_transform, random_state)
        self.nearest_sampler = KNearestPrototypeSampler(
            x, y, n_prototypes, metric_transform, random_state
        )
        self.shapelet_sampler = ShapeletPrototypeSampler(
            x,
            y,
            n_prototypes,
            metric_transform,
            random_state,
            min_shapelet_size,
            max_shapelet_size,
        )

    def sample(self, o):
        p = self.nearest_sampler.nearest_index(o)
        return self.shapelet_sampler.sample_shapelet(self.x[p])

    def move(self, o, p):
        return self.shapelet_sampler.move(o, p)


class MetricTransform(abc.ABC):
    """Move a time series towards a prototype"""

    def __init__(self, gamma):
        """Construct a new transformer

        Parameters
        ----------
        gamma : float
            The strength of the move, where values close to 0 means that the
            sample is moved less and values closer to 1 mean that the sample
            is moved more.
        """
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1], got %r" % gamma)
        self.gamma = gamma

    @abc.abstractmethod
    def move(self, o, p):
        """Move the sample o towards p

        Parameters
        ----------
        o : ndarray of shape (n_timestep,)
            An array
        p : ndarray of shape (n_timestep,)
            An array

        Returns
        -------
        ndarray : an array
        """
        pass


class EuclideanTransform(MetricTransform):
    """Transform a sample by moving it closer in euclidean space"""

    def move(self, o, p):
        return ((1 - self.gamma) * o) + self.gamma * p


class DynamicTimeWarpTransform(MetricTransform):
    """Transform a sample by moving it closer using the dtw alignment matrix"""

    def __init__(self, gamma, r=1.0):
        super().__init__(gamma)
        self.r = r

    def move(self, o, p):
        indicator, indices = dtw_mapping(o, p, r=self.r, return_index=True)
        result = o * (1 - self.gamma)
        weight = np.ones(o.shape[0]) * (1 - self.gamma)
        for i, j in zip(*indices):
            result[j] += self.gamma * p[j]
            weight[j] += self.gamma
        return result / weight


_METRIC_TRANSFORM = {
    "euclidean": EuclideanTransform,
    "dtw": DynamicTimeWarpTransform,
}

_PROTOTYPE_SAMPLER = {
    "sample": UniformPrototypeSampler,
    "nearest": KNearestPrototypeSampler,
    "shapelet": ShapeletPrototypeSampler,
    "nearest_shapelet": KNearestShapeletPrototypeSampler,
}
