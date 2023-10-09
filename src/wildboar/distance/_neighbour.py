import numbers

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils._param_validation import Interval

from ..base import BaseEstimator
from ._distance import pairwise_distance


class KNeighbourClassifier(ClassifierMixin, BaseEstimator):
    """
    Classifier implementing k-nearest neighbours

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
        "n_neighbours": Interval(numbers.Integral, 1, None, closed="left"),
        "metric": str,
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
        ndarray of shape (n_samples, len(self.classes_))
            The probability of each class for each sample.
        """
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
        return np.take(self.classes_, np.argmax(self.predict_proba(x), axis=1))


class _KMeansCluster:
    def __init__(self, init, metric, metric_params, random_state):
        self.metric = metric
        self.metric_params = metric_params
        self.centroids_ = init
        self.random_state = random_state

    def cost(self, x):
        if not hasattr(self, "_assigned") or self._assigned is None:
            raise ValueError("`assign` must be called before `cost`")

        return (
            paired_distance(
                x,
                self.centroids_[self._assigned],
                dim="mean",
                metric=self.metric,
                metric_params=self.metric_params,
            ).sum()
            / x.shape[0]
        )

    def assign(self, x):
        self._assigned = pairwise_distance(
            x,
            self.centroids_,
            dim="mean",
            metric=self.metric,
            metric_params=self.metric_params,
        ).argmin(axis=1)

        for c in range(self.centroids_.shape[0]):
            cluster = x[self._assigned == c]
            if cluster.shape[0] == 0:
                rnd = self.random_state.randint(x.shape[0])
                self._assigned[rnd] = c
                self.centroids_[c] = x[rnd]  # FIXME!
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
            method="ssg",
            random_state=self.random_state.randint(np.iinfo(np.int32).max),
        )


class KMeans(ClusterMixin, BaseEstimator):
    _parameter_constraints: dict = {
        "n_clusters": [Interval(numbers.Integral, 2, None, closed="left")],
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
        tol=1e-4,
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

    def fit(self, x, y=None, sample_weight=None):
        self._validate_params()
        x = self._validate_data(x, allow_3d=False)
        random_state = check_random_state(self.random_state)
        centroids = x[random_state.randint(x.shape[0], size=self.n_clusters)]

        if self.metric == "euclidean":
            cluster = _EuclideanCluster(centroids, random_state=random_state)
        elif self.metric == "dtw":
            if self.g is None:
                cluster = _DtwCluster(centroids, r=self.r, random_state=random_state)
            else:
                cluster = _WDtwCluster(
                    centroids, r=self.r, g=self.g, random_state=random_state
                )

        prev_cost = np.inf
        cost = -np.inf
        iter = 0
        while iter < self.max_iter and not np.isclose(cost, prev_cost, atol=self.tol):
            cluster.assign(x)
            prev_cost, cost = cost, cluster.cost(x)
            iter += 1

            if self.verbose > 0:
                print(f"Iteration {iter}, {cost} (prev_cost = {prev_cost})")

        self.n_iter_ = iter
        self.cluster_centers_ = cluster.centroids_
        self.labels_ = cluster._assigned
        return self

    def predict(self, x):
        pass
