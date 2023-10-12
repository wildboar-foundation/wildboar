# Authors: Isak Samsten
# License: BSD 3 clause
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.cluster import KMeans as Sklearn_KMeans
from sklearn.metrics.pairwise import pairwise_distances as sklearn_pairwise_distances
from sklearn.neighbors import KNeighborsClassifier as Sklearn_KNeighborsClassifier
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.validation import check_is_fitted, check_random_state

from ...base import BaseEstimator, CounterfactualMixin, ExplainerMixin
from ...distance._distance import pairwise_distance
from ...distance._neighbors import KMeans, KMedoids


class _Explainer(metaclass=ABCMeta):
    def __init__(self, n_neighbors, n_cluster, metric, metric_params, random_state):
        self.n_clusters = n_cluster
        self.metric = metric
        self.metric_params = metric_params
        self.random_state = random_state
        self.n_neighbors = n_neighbors

    @abstractmethod
    def _cluster(self, x):
        pass

    @abstractmethod
    def _pairwise_distance(self, x, y):
        pass

    def _assign(self, x, y, classes):
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("_cluster must be called before _assign")

        n_classes = len(classes)
        self.majority_centers_ = {}

        # Find top n_neighbors labels from the traning samples
        # for the cluster centers
        cluster_labels = np.take(
            y,
            np.argpartition(
                self._pairwise_distance(self.cluster_centers_, x),
                self.n_neighbors,
                axis=1,
            )[:, : self.n_neighbors],
        )

        for cls in classes:
            cls_idx = np.nonzero(classes == cls)[0][0]
            # Store the cluster centers that would be predicted as cls
            self.majority_centers_[cls] = self.cluster_centers_[
                (cluster_labels == cls_idx).sum(axis=1) >= self.n_neighbors // 2 + 1
            ]

    def _find_closest_center(self, x, y):
        if not hasattr(self, "majority_centers_"):
            raise ValueError("_assign must be called before _find_closest_center")

        centers = self.majority_centers_[y]
        if centers.shape[0] == 0:
            return None

        return centers[
            self._pairwise_distance(x, centers).argmin(axis=1),
            :,
        ]


class _SklearnExplainer(_Explainer):
    def _pairwise_distance(self, x, y):
        metric_params = self.metric_params if self.metric_params is not None else {}
        return sklearn_pairwise_distances(x, y, metric=self.metric, **metric_params)


class _SklearnKMeansExplainer(_SklearnExplainer):
    def _cluster(self, x):
        kmeans = Sklearn_KMeans(
            n_clusters=self.n_clusters, n_init=10, random_state=self.random_state
        ).fit(x)

        self.cluster_centers_ = kmeans.cluster_centers_
        self.labels_ = kmeans.labels_


class _SklearnKMedoidsExplainer(_SklearnExplainer):
    def _cluster(self, x):
        metric_params = self.metric_params if self.metric_params is not None else {}
        kmedoids = KMedoids(
            n_clusters=self.n_clusters,
            metric="precomputed",
            metric_params=self.metric_params,
            random_state=self.random_state,
        ).fit(sklearn_pairwise_distances(x, metric=self.metric, **metric_params))

        self.cluster_centers_ = x[kmedoids.medoid_indices_, :]
        self.labels_ = kmedoids.labels_


class _WildboarExplainer(_Explainer):
    def _pairwise_distance(self, x, y):
        return pairwise_distance(
            x,
            y,
            metric=self.metric,
            metric_params=self.metric_params,
            dim="mean",
        )


class _WildboarKMeansExplainer(_WildboarExplainer):
    def _cluster(self, x):
        metric = self.metric
        if self.metric_params is not None:
            r = self.metric_params.get("r", 1.0)
            g = self.metric_params.get("g")
        else:
            r = 1.0
            g = None

        if self.metric == "wdtw":
            metric = "dtw"

        kmeans = KMeans(
            n_clusters=self.n_clusters,
            metric=metric,
            r=r,
            g=g,
            random_state=self.random_state,
        ).fit(x)
        self.cluster_centers_ = kmeans.cluster_centers_
        self.labels_ = kmeans.labels_


class _WildboarKMedoidsExplainer(_WildboarExplainer):
    def _cluster(self, x):
        kmedoids = KMedoids(
            n_clusters=self.n_clusters,
            metric=self.metric,
            metric_params=self.metric_params,
            random_state=self.random_state,
        ).fit(x)
        self.cluster_centers_ = kmedoids.cluster_centers_
        self.labels_ = kmedoids.labels_


class KNeighborsCounterfactual(CounterfactualMixin, ExplainerMixin, BaseEstimator):
    """
    Fit a counterfactual explainer to a k-nearest neighbors classifier.

    Parameters
    ----------
    method : {"auto", "mean", "medoid"}, optional
        The method for generating counterfactuals. If 'auto', counterfactuals
        are generated using k-means if possible and k-medoids otherwise. If
        'mean', counterfactuals are always generated using k-means, which fails
        if the estimator is fitted with a metric other than 'euclidean', 'dtw'
        or 'wdtw. If 'medoid', counterfactuals are generated using k-medoids.

        .. versionadded:: 1.2
    random_state : int or RandomState, optional
        - If `int`, `random_state` is the seed used by the random number generator.
        - If `RandomState` instance, `random_state` is the random number generator.
        - If `None`, the random number generator is the `RandomState` instance used
          by `np.random`.

    Attributes
    ----------
    explainer_ : dict
        The explainer for each label

    References
    ----------
    Karlsson, I., Rebane, J., Papapetrou, P., & Gionis, A. (2020).
        Locally and globally explainable time series tweaking.
        Knowledge and Information Systems, 62(5), 1671-1700.
    """

    _parameter_constraints: dict = {
        "method": [StrOptions({"auto", "mean", "medoid"})],
        "random_state": ["random_state"],
    }

    def __init__(self, method="auto", random_state=None):
        self.random_state = random_state
        self.method = method

    def _validate_estimator(self, estimator, allow_3d=False):
        estimator = super()._validate_estimator(estimator, allow_3d)
        for attr in ["_fit_X", "_y", "n_neighbors", "classes_", "metric"]:
            if not hasattr(estimator, attr):
                raise ValueError(f"estimator must be have the attributes {attr}.")

        return estimator

    def fit(self, estimator, x=None, y=None):
        self._validate_params()
        estimator = self._validate_estimator(estimator)
        x = estimator._fit_X
        y = estimator._y
        self.n_timesteps_in_ = estimator.n_features_in_
        self.n_dims_in_ = 1
        self.n_features_in_ = self.n_timesteps_in_

        n_clusters = x.shape[0] // estimator.n_neighbors

        random_state = check_random_state(self.random_state)

        if isinstance(estimator, Sklearn_KNeighborsClassifier):
            if (
                estimator.metric == "euclidean"
                or (estimator.p == 2 and estimator.metric == "minkowski")
            ) and self.method in ("auto", "mean"):
                Explainer = _SklearnKMeansExplainer
            elif self.method in ("auto", "medoid"):
                Explainer = _SklearnKMedoidsExplainer
            else:
                raise ValueError(
                    (
                        "if method='mean', estimator %r must be fitted with "
                        "metric 'euclidean' or 'minkowski' with 'p=1', got %r"
                    )
                    % (estimator.__class__.__qualname__, estimator.metric)
                )
        elif estimator.metric in ("euclidean", "dtw", "wdtw") and self.method in (
            "auto",
            "mean",
        ):
            Explainer = _WildboarKMeansExplainer
        elif self.method in ("auto", "medoid"):
            Explainer = _WildboarKMedoidsExplainer
        else:
            raise ValueError(
                (
                    "if method='mean', estimator %r must be fitted with metric in "
                    "'euclidean', 'dtw' or 'wdtw', got %r"
                )
                % (estimator.__class__.__qualname__, estimator.metric)
            )

        self.explainer_ = Explainer(
            n_neighbors=estimator.n_neighbors,
            n_cluster=n_clusters,
            metric=estimator.metric,
            metric_params=estimator.metric_params
            if hasattr(estimator, "metric_params")
            else None,
            random_state=random_state,
        )

        self.explainer_._cluster(x)
        self.explainer_._assign(x, y, estimator.classes_)
        return self

    def explain(self, x, y):
        check_is_fitted(self)
        x, y = self._validate_data(x, y, reset=False, dtype=float)
        x_counterfactuals = x.copy()
        labels = np.unique(y)
        for label in labels:
            label_indices = np.where(y == label)[0]

            closest = self.explainer_._find_closest_center(x[label_indices], label)
            if closest is not None:
                x_counterfactuals[label_indices, :] = closest

        return x_counterfactuals

    def _more_tags():
        return {"X_types": []}
