import math
import numbers

import numpy as np
from sklearn.base import ClassifierMixin, ClusterMixin, TransformerMixin
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, check_random_state

from ..base import BaseEstimator
from ..utils.validation import check_classification_targets
from ._distance import paired_distance, pairwise_distance
from .dtw import dtw_average


class KNeighbourClassifier(ClassifierMixin, BaseEstimator):
    """
    Classifier implementing k-nearest neighbours.

    Parameters
    ----------
    n_neighbours : int, optional
        The number of neighbours.
    metric : str, optional
        The distance metric.
    metric_params : dict, optional
        Optional parameters to the distance metric.

        Read more about the metrics and their parameters in the
        :ref:`User guide <list_of_metrics>`.

    Attributes
    ----------
    classes_ : ndarray of shapel (n_classes, )
        Known class labels.
    """

    _parameter_constraints: dict = {
        "n_neighbours": [Interval(numbers.Integral, 1, None, closed="left")],
        "metric": [str],
        "metric_params": [str, None],
    }

    def __init__(self, n_neighbours=5, *, metric="euclidean", metric_params=None):
        self.n_neighbours = n_neighbours
        self.metric = metric
        self.metric_params = metric_params

    def fit(self, x, y):
        """
        Fit the classifier to the training data.

        Parameters
        ----------
        x : univariate time-series or multivaraite time-series
            The input samples.
        y : array-like of shape (n_samples, )
            The input labels.

        Returns
        -------
        KNeighbourClassifier
            This instance.
        """
        self._validate_params()
        x, y = self._validate_data(x, y, allow_3d=True)
        check_classification_targets(y)
        self._fit_X = x.copy()  # Align naming with sklearn
        self._y = y.copy()
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, x):
        """
        Compute probability estimates for the samples in x.

        Parameters
        ----------
        x : univariate time-series or multivariate time-series
            The input samples.

        Returns
        -------
        ndarray of shape (n_samples, len(self.classes\_))
            The probability of each class for each sample.
        """
        check_is_fitted(self)
        x = self._validate_data(x, allow_3d=True, reset=False)
        dists = pairwise_distance(
            x,
            self._fit_X,
            dim="mean",
            metric=self.metric,
            metric_params=self.metric_params,
        )
        preds = self._y[
            np.argpartition(dists, self.n_neighbours, axis=1)[:, : self.n_neighbours]
        ]

        probs = np.empty((x.shape[0], len(self.classes_)), dtype=float)
        for i, c in enumerate(self.classes_):
            probs[:, i] = np.sum(preds == c, axis=1) / self.n_neighbours
        return probs

    def predict(self, x):
        """
        Compute the class label for the samples in x.

        Parameters
        ----------
        x : univariate time-series or multivariate time-series
            The input samples.

        Returns
        -------
        ndarray of shape (n_samples, )
            The class label for each sample.
        """
        proba = np.argmax(self.predict_proba(x), axis=1)
        return np.take(self.classes_, proba)


class _KMeansCluster:
    def __init__(self, init, metric, metric_params, random_state):
        self.metric = metric
        self.metric_params = metric_params
        self.centroids_ = init
        self.random_state = random_state

    def cost(self, x):
        if not hasattr(self, "assigned_") or self.assigned_ is None:
            raise ValueError("`assign` must be called before `cost`")

        return (
            paired_distance(
                x,
                self.centroids_[self.assigned_],
                dim="mean",
                metric=self.metric,
                metric_params=self.metric_params,
            ).sum()
            / x.shape[0]
        )

    def assign(self, x):
        self.distance_ = pairwise_distance(
            x,
            self.centroids_,
            dim="mean",
            metric=self.metric,
            metric_params=self.metric_params,
        )
        self.assigned_ = self.distance_.argmin(axis=1)

    def update(self, x):
        for c in range(self.centroids_.shape[0]):
            cluster = x[self.assigned_ == c]
            if cluster.shape[0] == 0:
                far_from_center = self.distance_.min(axis=1).argmax()
                self.assigned_[far_from_center] = c
                self.centroids_[c] = x[far_from_center]
            elif cluster.shape[0] == 1:
                self.centroids_[c] = cluster
            else:
                self.centroids_[c] = self._update_centroid(cluster, self.centroids_[c])


class _EuclideanCluster(_KMeansCluster):
    def __init__(self, centroids, random_state):
        super().__init__(centroids, "euclidean", None, random_state)

    def _update_centroid(self, group, _centroid):
        return group.mean(axis=0)


class _DtwCluster(_KMeansCluster):
    def __init__(self, centroids, r, random_state):
        super().__init__(centroids, "dtw", {"r": r}, random_state)
        self.r = r

    def _update_centroid(self, group, current_center):
        return dtw_average(
            group,
            r=self.r,
            init=current_center,
            method="mm",
            random_state=self.random_state.randint(np.iinfo(np.int32).max),
        )


class _WDtwCluster(_KMeansCluster):
    def __init__(self, centroids, r, g, random_state):
        super().__init__(centroids, "wdtw", {"r": r, "g": g}, random_state)
        self.r = r
        self.g = g

    def _update_centroid(self, group, current_center):
        return dtw_average(
            group,
            r=self.r,
            g=self.g,
            init=current_center,
            method="mm",
            random_state=self.random_state.randint(np.iinfo(np.int32).max),
        )


class KMeans(ClusterMixin, TransformerMixin, BaseEstimator):  # noqa: I0012
    """
    KMeans clustering with support for DTW and weighted DTW.

    Parameters
    ----------
    n_clusters : int, optional
        The number of clusters.
    metric : {"euclidean", "dtw"}, optional
        The metric.
    r : float, optional
        The size of the warping window.
    g : float, optional
        SoftDTW penalty. If None, traditional DTW is used.
    init : {"random"}, optional
        Cluster initialization. If "random", randomly initialize `n_clusters`.
    n_init : "auto" or int, optional
        Number times the algorithm is re-initialized with new centroids.
    max_iter : int, optional
        The maximum number of iterations for a single run of the algorithm.
    tol : float, optional
        Relative tolerance to declare convergence of two consecutive iterations.
    verbose : int, optional
        Print diagnostic messages during convergence.
    random_state : RandomState or int, optional
        Determines random number generation for centroid initialization and
        barycentering when fitting with `metric="dtw"`.

    Attributes
    ----------
    n_iter_ : int
        The number of iterations before convergence.
    cluster_centers_ : ndarray of shape (n_clusters, n_timestep)
        The cluster centers.
    labels_ : ndarray of shape (n_samples, )
        The cluster assignment.
    """

    _parameter_constraints: dict = {
        "n_clusters": [Interval(numbers.Integral, 1, None, closed="left")],
        "metric": [StrOptions({"euclidean", "dtw"})],
        "r": [Interval(numbers.Real, 0, 1, closed="both")],
        "g": [None, Interval(numbers.Real, 0, None, closed="neither")],
        "init": [StrOptions({"random"})],
        "n_init": [
            StrOptions({"auto"}),
            Interval(numbers.Integral, 1, None, closed="left"),
        ],
        "max_iter": [Interval(numbers.Integral, 1, None, closed="left")],
        "tol": [float],
        "verbose": [int],
        "random_state": ["random_state", None],
    }

    def __init__(
        self,
        n_clusters=8,
        *,
        metric="euclidean",
        r=1.0,
        g=None,
        init="random",
        n_init="auto",
        max_iter=300,
        tol=1e-3,
        verbose=0,
        random_state=None,
    ):
        self.metric = metric
        self.r = r
        self.g = g
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, x, y=None):
        """
        Compute the kmeans-clustering.

        Parameters
        ----------
        x : univariate time-series
            The input samples.
        y : Ignored, optional
            Not used.

        Returns
        -------
        object
            Fitted estimator.
        """
        self._validate_params()
        x = self._validate_data(x, allow_3d=False)
        if self.n_init == "auto":
            n_init = 1
        else:
            n_init = self.n_init

        random_state = check_random_state(self.random_state)
        best_cost = np.inf
        best_cluster = None
        best_reassign = True
        best_iter = None
        for init in range(n_init):
            iter, cost, cluster, reassign = self._fit_one_init(x, random_state)
            if cost < best_cost:
                best_cost = cost
                best_cluster = cluster
                best_reassign = reassign
                best_iter = iter

        self.n_iter_ = best_iter
        self.inertia_ = best_cost
        self.cluster_centers_ = best_cluster.centroids_

        if best_reassign:
            best_cluster.assign(x)

        self.labels_ = best_cluster.assigned_
        return self

    def _fit_one_init(self, x, random_state):
        if self.init == "random":
            centroids = x[
                random_state.choice(x.shape[0], size=self.n_clusters, replace=False)
            ]
        else:
            raise ValueError(f"Unsupported 'init', got {self.init}")

        if self.metric == "euclidean":
            cluster = _EuclideanCluster(centroids, random_state=random_state)
        elif self.metric == "dtw":
            if self.g is None:
                cluster = _DtwCluster(centroids, r=self.r, random_state=random_state)
            else:
                cluster = _WDtwCluster(
                    centroids, r=self.r, g=self.g, random_state=random_state
                )
        else:
            raise ValueError(f"Unsupported 'metric', got {self.metric}")

        prev_cost = np.inf
        cost = -np.inf
        reassign = True
        for iter in range(self.max_iter):
            cluster.assign(x)
            prev_cost, cost = cost, cluster.cost(x)
            if self.verbose > 0:
                print(f"Iteration {iter}, {cost} (prev_cost = {prev_cost})")

            if math.isclose(cost, prev_cost, rel_tol=self.tol):
                reassign = False
                break

            cluster.update(x)

        return iter, cost, cluster, reassign

    def predict(self, x):
        """
        Predict the closest cluster for each sample.

        Parameters
        ----------
        x : univariate time-series
            The input samples.

        Returns
        -------
        ndarray of shape (n_samples, )
            Index of the cluster each sample belongs to.
        """
        return self.transform(x).argmin(axis=1)

    def transform(self, x):
        """
        Transform the input to a cluster distance space.

        Parameters
        ----------
        x : univariate time-series
            The input samples.

        Returns
        -------
        ndarray of shape (n_samples, n_clusters)
            The distance between each sample and each cluster.
        """
        check_is_fitted(self)
        self._validate_data(x, allow_3d=False, reset=False)
        metric_params = {}
        metric = self.metric
        if self.metric == "dtw":
            metric_params["r"] = self.r
            if self.g is not None:
                metric_params["g"] = self.g
                metric = "wdtw"

        return pairwise_distance(
            x,
            self.cluster_centers_,
            dim="mean",
            metric=metric,
            metric_params=metric_params,
        )
