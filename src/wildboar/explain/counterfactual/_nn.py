# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.utils.validation import check_is_fitted

from ...base import BaseEstimator, CounterfactualMixin, ExplainerMixin


class KNeighborsCounterfactual(CounterfactualMixin, ExplainerMixin, BaseEstimator):
    """Fit a counterfactual explainer to a k-nearest neighbors classifier

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

    def __init__(self, random_state=None):
        self.random_state = random_state

    def _validate_estimator(self, estimator, allow_3d=False):
        if not isinstance(estimator, KNeighborsClassifier):
            raise ValueError("not a valid estimator")

        if estimator.metric == "euclidean" or (
            estimator.metric == "minkowski" and estimator.p == 2
        ):
            return super()._validate_estimator(estimator, allow_3d)
        else:
            raise ValueError(
                "estimator must be fit with metric='euclidean', got %r"
                % estimator.metric
            )

    def fit(self, estimator, x=None, y=None):
        estimator = self._validate_estimator(estimator)
        x = estimator._fit_X
        y = estimator._y

        self.n_timesteps_in_ = estimator.n_features_in_
        self.n_dims_in_ = 1
        self.n_features_in_ = self.n_timesteps_in_

        classes = estimator.classes_
        n_clusters = x.shape[0] // estimator.n_neighbors
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state).fit(x)
        n_classes = len(classes)

        label_nn = {}
        for cls in classes:
            to_idx = np.nonzero(classes == cls)[0][0]
            center_majority = np.zeros([kmeans.n_clusters, n_classes])
            for cluster_label, class_label in zip(kmeans.labels_, y):
                center_majority[cluster_label, class_label] += 1

            center_prob = center_majority / np.sum(center_majority, axis=1).reshape(
                -1, 1
            )
            majority_class = center_prob[:, to_idx] > (1.0 / n_classes)
            maximum_class = (
                center_majority[:, to_idx] >= (estimator.n_neighbors // n_classes) + 1
            )
            cluster_centers = kmeans.cluster_centers_
            majority_centers = cluster_centers[majority_class & maximum_class, :]
            majority_center_nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
            if majority_centers.shape[0] > 0:
                majority_center_nn.fit(majority_centers)

            label_nn[cls] = (majority_center_nn, majority_centers)
        self.explainer_ = label_nn
        return self

    def explain(self, x, y):
        check_is_fitted(self)
        x, y = self._validate_data(x, y, reset=False, dtype=float)
        x_counterfactuals = x.copy()
        labels = np.unique(y)
        for label in labels:
            label_indices = np.where(y == label)[0]

            nn, mc = self.explainer_[label]
            if mc.shape[0] > 0:
                closest = nn.kneighbors(x[label_indices, :], return_distance=False)
                x_counterfactuals[label_indices, :] = mc[closest[:, 0], :]

        return x_counterfactuals

    def _more_tags():
        return {"X_types": []}
