import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.utils.validation import check_is_fitted

from .base import BaseCounterfactual


class KNeighborsCounterfactual(BaseCounterfactual):
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

    Karlsson, I., Rebane, J., Papapetrou, P., & Gionis, A. (2018).
        Explainable time series tweaking via irreversible and reversible temporal
        transformations. In 2018 IEEE International Conference on Data Mining (ICDM)
    """

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, estimator):
        if not isinstance(estimator, KNeighborsClassifier):
            raise ValueError("not a valid estimator")
        check_is_fitted(estimator)
        if estimator.metric != "euclidean":
            raise ValueError("only euclidean distance is supported, got %r" % estimator.metric)

        x = estimator._fit_X
        y = estimator._y
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

            center_prob = center_majority / np.sum(center_majority, axis=1).reshape(-1, 1)
            majority_class = center_prob[:, to_idx] > (1.0 / n_classes)
            maximum_class = center_majority[:, to_idx] >= (estimator.n_neighbors // n_classes) + 1
            cluster_centers = kmeans.cluster_centers_
            majority_centers = cluster_centers[majority_class & maximum_class, :]
            majority_center_nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
            if majority_centers.shape[0] > 0:
                majority_center_nn.fit(majority_centers)

            label_nn[cls] = (majority_center_nn, majority_centers)
        self.explainer_ = label_nn
        return self

    def transform(self, x, y):
        check_is_fitted(self, ["explainer_"])
        x_counter = np.empty(x.shape, dtype=x.dtype)
        success = np.empty(x.shape[0], dtype=bool)
        labels = np.unique(y)
        for label in labels:
            label_indices = np.where(y == label)[0]

            nn, mc = self.explainer_[label]
            if mc.shape[0] > 0:
                closest = nn.kneighbors(x[label_indices, :], return_distance=False)
                x_counter[label_indices, :] = mc[closest[:, 0], :]
                success[label_indices] = True
            else:
                success[label_indices] = False
        return x_counter, success
