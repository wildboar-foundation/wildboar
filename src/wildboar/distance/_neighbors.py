import math
import numbers
import warnings

import numpy as np
from sklearn.base import ClassifierMixin, ClusterMixin, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, check_random_state

from ..base import BaseEstimator
from ..utils.validation import check_classification_targets
from ._cneighbors import _pam_build, _pam_optimal_swap
from ._distance import _METRICS, argmin_distance, paired_distance, pairwise_distance
from .dtw import dtw_average


class KNeighborsClassifier(ClassifierMixin, BaseEstimator):
    """
    Classifier implementing k-nearest neighbors.

    Parameters
    ----------
    n_neighbors : int, optional
        The number of neighbors.
    metric : str, optional
        The distance metric.
    metric_params : dict, optional
        Optional parameters to the distance metric.

        Read more about the metrics and their parameters in the
        :ref:`User guide <list_of_metrics>`.
    n_jobs : int, optional
        The number of parallel jobs.

    Attributes
    ----------
    classes_ : ndarray of shapel (n_classes, )
        Known class labels.
    """

    _parameter_constraints: dict = {
        "n_neighbors": [Interval(numbers.Integral, 1, None, closed="left")],
        "metric": [StrOptions(_METRICS.keys())],
        "metric_params": [dict, None],
        "n_jobs": [numbers.Integral, None],
    }

    def __init__(
        self,
        n_neighbors=5,
        *,
        metric="euclidean",
        metric_params=None,
        n_jobs=None,
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

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
        KNeighborClassifier
            This instance.
        """
        self._validate_params()
        x, y = self._validate_data(x, y, allow_3d=True)
        check_classification_targets(y)
        self._fit_X = x.copy()  # Align naming with sklearn
        self.classes_, self._y = np.unique(y, return_inverse=True)
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
        ndarray of shape (n_samples, len(self.classes_))
            The probability of each class for each sample.
        """
        check_is_fitted(self)
        x = self._validate_data(x, allow_3d=True, reset=False)

        # Treat a multivariate time series with a single dimension as a
        # univariate time series to ensure that we use the fast path.
        if x.ndim == 3 and x.shape[1] == 1:
            x = x.reshape(x.shape[0], -1)

        if x.ndim == 3:
            dists = pairwise_distance(
                x,
                self._fit_X,
                dim="mean",
                metric=self.metric,
                metric_params=self.metric_params,
                n_jobs=self.n_jobs,
            )

            closest = np.argpartition(dists, self.n_neighbors, axis=1)[
                :, : self.n_neighbors
            ]
        else:
            closest = argmin_distance(
                x,
                self._fit_X,
                metric=self.metric,
                metric_params=self.metric_params,
                k=self.n_neighbors,
                n_jobs=self.n_jobs,
            )

        preds = self._y[closest]
        probs = np.empty((x.shape[0], len(self.classes_)), dtype=float)
        for i in range(len(self.classes_)):
            probs[:, i] = np.sum(preds == i, axis=1) / self.n_neighbors
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
        if x.ndim == 2:
            self.assigned_, self.distance_ = argmin_distance(
                x,
                self.centroids_,
                k=1,
                metric=self.metric,
                metric_params=self.metric_params,
                return_distance=True,
            )
            self.assigned_ = np.ravel(self.assigned_)
        else:
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


class _KMedoidsCluster:
    def __init__(self, dist, cluster_idx):
        self._dist = dist
        self._cluster_idx = cluster_idx
        self._prev_cluster_idx = cluster_idx.copy()

    def assign(self):
        self._prev_cluster_idx[:] = self._cluster_idx
        self.labels_ = self._dist[self._cluster_idx].argmin(axis=0)

    def cost(self):
        self.cost_ = (
            np.take(self._dist, self._cluster_idx[self.labels_]).sum()
            / self._dist.shape[0]
        )  # np.sum(self._prev_cluster_idx != self._cluster_idx)

    def update(self):
        pass


class _FastKMedoidsCluster(_KMedoidsCluster):
    def update(self):
        for idx in range(self._cluster_idx.shape[0]):
            cluster_idx = np.where(self.labels_ == idx)[0]
            if cluster_idx.shape[0] == 0:
                continue

            cost = self._dist[cluster_idx, cluster_idx.reshape(-1, 1)].sum(axis=1)
            min_idx = cost.argmin()
            min_cost = cost[min_idx]
            curr_cost = cost[(cluster_idx == self._cluster_idx[idx]).argmax()]
            if min_cost < curr_cost:
                self._cluster_idx[idx] = cluster_idx[min_idx]


class _PamKMedoidsCluster(_KMedoidsCluster):
    def __init__(self, dist, cluster_idx):
        super().__init__(dist, cluster_idx)
        self._djs, self._ejs = np.sort(self._dist[self._cluster_idx], axis=0)[[0, 1]]

    def update(self):
        not_cluster_idx = np.delete(np.arange(self._dist.shape[0]), self._cluster_idx)
        optimal_swap = _pam_optimal_swap(
            self._dist,
            self._cluster_idx,
            not_cluster_idx,
            self._djs,
            self._ejs,
            self._cluster_idx.shape[0],
        )

        if optimal_swap is not None:
            i, j, _ = optimal_swap
            self._cluster_idx[self._cluster_idx == i] = j
            self._djs, self._ejs = np.sort(self._dist[self._cluster_idx], axis=0)[
                [0, 1]
            ]


_KMEDOID_CLUSTER_ALGORITHMS = {"pam": _PamKMedoidsCluster, "fast": _FastKMedoidsCluster}


class KMedoids(ClusterMixin, TransformerMixin, BaseEstimator):
    """
    KMedoid algorithm.

    Parameters
    ----------
    n_clusters : int, optional
        The number of clusters.
    metric : str, optional
        The metric.
    metric_params : dict, optional
        The metric parameters. Read more about the metrics and their parameters
        in the :ref:`User guide <list_of_metrics>`.
    init : {"auto", "random", "min"}, optional
        Cluster initialization. If "random", randomly initialize `n_clusters`,
        if "min" select the samples with the smallest distance to the other samples.
    n_init : "auto" or int, optional
        Number times the algorithm is re-initialized with new centroids.
    algorithm : {"fast", "pam"}, optional
        The algorithm for updating cluster assignments. If "pam", use the
        Partitioning Around Medoids algorithm.
    max_iter : int, optional
        The maximum number of iterations for a single run of the algorithm.
    tol : float, optional
        Relative tolerance to declare convergence of two consecutive iterations.
    verbose : int, optional
        Print diagnostic messages during convergence.
    n_jobs : int, optional
        The number of jobs to run in parallel. A value of `None` means using
        a single core and a value of `-1` means using all cores. Positive
        integers mean the exact number of cores.
    random_state : RandomState or int, optional
        Determines random number generation for centroid initialization and
        barycentering when fitting with `metric="dtw"`.

    Attributes
    ----------
    n_iter_ : int
        The number of iterations before convergence.
    cluster_centers_ : ndarray of shape (n_clusters, n_timestep)
        The cluster centers.
    medoid_indices_ : ndarray of shape (n_clusters, )
        The index of the medoid in the input samples.
    labels_ : ndarray of shape (n_samples, )
        The cluster assignment.
    """

    _parameter_constraints: dict = {
        "n_clusters": [Interval(numbers.Integral, 1, None, closed="left")],
        "metric": [StrOptions(_METRICS.keys() | {"precomputed"})],
        "metric_params": [None, dict],
        "init": [StrOptions({"random", "auto", "min"})],
        "n_init": [
            StrOptions({"auto"}),
            Interval(numbers.Integral, 1, None, closed="left"),
        ],
        "algorithm": [StrOptions({"fast", "pam"})],
        "max_iter": [Interval(numbers.Integral, 1, None, closed="left")],
        "tol": [Interval(numbers.Real, 0, None, closed="left")],
        "verbose": [numbers.Integral],
        "n_jobs": [None, numbers.Integral],
        "random_state": ["random_state", None],
    }

    def __init__(
        self,
        n_clusters=8,
        metric="euclidean",
        metric_params=None,
        init="random",
        n_init="auto",
        algorithm="fast",
        max_iter=30,
        tol=1e-4,
        verbose=0,
        n_jobs=None,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.metric = metric
        self.metric_params = metric_params
        self.init = init
        self.n_init = n_init
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, x, y=None):
        """
        Compute the kmedoids-clustering.

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
        x = self._validate_data(x, allow_3d=True)
        self._validate_params()

        if self.n_init == "auto":
            if self.init == "auto":
                n_init = 1
            else:
                n_init = 10
        else:
            n_init = self.n_init

        # initial medoids are deterministic
        if n_init > 1 and self.init != "random":
            n_init = 1

        max_iter = self.max_iter
        if self.algorithm == "pam" and self.n_clusters == 1 and self.max_iter != 0:
            warnings.warn("n_clusters must be larger than 1 if max_iter larger than 0")
            max_iter = 0

        random_state = check_random_state(self.random_state)
        if self.metric == "precomputed":
            dist = x
        else:
            dist = pairwise_distance(
                x,
                dim="mean",
                metric=self.metric,
                metric_params=self.metric_params,
                n_jobs=self.n_jobs,
            )
        best_iter = 0
        best_cost = np.inf
        best_clusterer = None
        best_reassign = None

        # TODO: Add sampling
        for i in range(n_init):
            if self.verbose:
                print(f"Running initialization {i}/{n_init}")

            iter, clusterer, reassign = self._fit_one_init(
                dist, random_state=random_state, max_iter=max_iter
            )

            if clusterer.cost_ < best_cost:
                best_cost = clusterer.cost_
                best_clusterer = clusterer
                best_iter = iter
                best_reassign = reassign

        if best_reassign:
            clusterer.assign()

        if self.metric == "precomputed":
            self.cluster_centers_ = None
        else:
            self.cluster_centers_ = x[best_clusterer._cluster_idx]

        self.inertia_ = best_clusterer.cost_
        self.medoid_indices_ = best_clusterer._cluster_idx
        self.n_iter_ = best_iter
        self.labels_ = best_clusterer.labels_
        return self

    def _fit_one_init(self, dist, *, max_iter, random_state):
        centers = self._init_centers(dist, self.n_clusters, random_state=random_state)
        clusterer = _KMEDOID_CLUSTER_ALGORITHMS[self.algorithm](dist, centers)
        clusterer.assign()
        reassign = False
        prev_cost = np.inf
        for iter in range(max_iter):
            clusterer.update()
            clusterer.cost()

            if self.verbose:
                print(
                    f"Iteration {iter}/{self.max_iter}: current_cost={clusterer.cost_}"
                )

            if math.isclose(clusterer.cost_, prev_cost, rel_tol=self.tol):
                reassign = True
                break

            prev_cost = clusterer.cost_
            clusterer.assign()

        if iter + 1 == self.max_iter:
            warnings.warn(
                "Maximum number of iterations reached before convergence. "
                "Consider increasing max_iter to improve the fit",
                ConvergenceWarning,
            )

        return iter, clusterer, reassign

    def _init_centers(self, dist, n_clusters, random_state):
        if self.init == "random":
            return random_state.choice(dist.shape[0], n_clusters, replace=False)
        elif self.init == "min" or (self.algorithm == "fast" and self.init == "auto"):
            return np.argpartition(dist.sum(axis=1), n_clusters)[:n_clusters]
        elif self.init == "auto":
            return _pam_build(dist, n_clusters)
        else:
            raise ValueError(f"Unsupported init, got {self.init}")

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
        x = self._validate_data(x, allow_3d=True, reset=False)
        if self.metric == "precomputed":
            return x[:, self.medoid_indices_]
        else:
            return pairwise_distance(
                x,
                self.cluster_centers_,
                dim="mean",
                metric=self.metric,
                metric_params=self.metric_params,
            )
