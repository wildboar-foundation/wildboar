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

import math
import numbers
import warnings

import numpy as np
from sklearn import clone
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state

from wildboar.linear_model import KernelLogisticRegression
from wildboar.utils import _soft_dependency_error, check_array

__all__ = [
    "kmeans_outliers",
    "density_outliers",
    "minority_outliers",
    "majority_outliers",
    "emmott_outliers",
]


def kmeans_outliers(
    x,
    y=None,
    *,
    n_outliers=0.05,
    n_clusters=5,
    random_state=None,
):
    """Label the samples of the cluster farthers from the other clusters as outliers.

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_timestep)
        The input samples

    y : ndarray of shape (n_samples, ), optional
        Ignored.

    n_outliers : float, optional
        The number of outlier samples expressed as a fraction of the inlier samples.

        - if float, the number of outliers are guaranteed but an error is raised
          if no cluster can satisfy the constraints. Lowering the ``n_cluster``
          parameter to allow for more samples per cluster.

    n_clusters : int, optional
        The number of clusters.

    random_state : int or RandomState
        - If `int`, `random_state` is the seed used by the random number generator
        - If `RandomState` instance, `random_state` is the random number generator
        - If `None`, the random number generator is the `RandomState` instance used
            by `np.random`.

    Returns
    -------
    x_outlier : ndarray of shape (n_inliers + n_outliers, n_timestep)
        The samples

    y_outlier : ndarray of shape (n_inliers + n_outliers, )
        The inliers (labeled as 1) and outlier (labled as -1)

    """
    x = check_array(x)
    random_state = check_random_state(random_state)
    k_means_ = KMeans(
        n_clusters=n_clusters, random_state=random_state.randint(np.iinfo(np.int32).max)
    )

    k_means_.fit(x)
    _, count = np.unique(k_means_.labels_, return_counts=True)
    min_sample = _min_samples(x.shape[0], n_outliers)

    invalid_clusters = np.where(count < min_sample)

    centroid_distance = pairwise_distances(
        k_means_.cluster_centers_, metric="euclidean"
    )

    # skip self matches and invalid clusters
    centroid_distance[centroid_distance == 0] = np.nan
    centroid_distance[invalid_clusters] = np.nan

    if np.all(np.isnan(centroid_distance)):
        raise ValueError("no valid clusters")

    # hide the warning for ignored clusters
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        outlier_cluster_ = np.nanargmax(np.nanmean(centroid_distance, axis=1))

    outlier_indices = np.where(k_means_.labels_ == outlier_cluster_)[0]
    inliers_indices = np.where(k_means_.labels_ != outlier_cluster_)[0]

    random_state.shuffle(outlier_indices)
    return _make_outlier_arrays(
        x,
        n_outliers=_n_outliers(
            n_outliers, outlier_indices.shape[0], inliers_indices.shape[0]
        ),
        outlier_indicies=outlier_indices,
        inlier_indicies=inliers_indices,
    )


def density_outliers(
    x,
    y=None,
    *,
    n_outliers=0.05,
    method="dbscan",
    eps=2.0,
    min_sample=5,
    metric="euclidean",
    max_eps=np.inf,
    random_state=None,
):
    """Labels samples as outliers if a density cluster algorithm fail to assign them to a
    cluster

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_timestep)
        The input samples

    y : ndarray of shape (n_samples, ), optional
        Ignored.

    n_outliers : float, optional
        The number of outlier samples expressed as a fraction of the inlier samples.
        By default, all samples of the minority class is considered as outliers.

    method : {"dbscan", "optics"}, optional
        The density based clustering method.

    eps : float, optional
        The eps parameter, when ``method="dbscan"``.

    min_sample : int, optional
        The ``min_sample`` parameter to the cluter method

    metric : str, optional
        The ``metric`` parameter to the cluster method

    max_eps : float, optional
        The ``max_eps`` parameter, when ``method="optics"``.

    Returns
    -------
    x_outlier : ndarray of shape (n_inliers + n_outliers, n_timestep)
        The samples

    y_outlier : ndarray of shape (n_inliers + n_outliers, )
        The inliers (labeled as 1) and outlier (labled as -1)

    """
    if method == "dbscan":
        estimator = DBSCAN(eps=eps, min_samples=min_sample, metric=metric)
    elif method == "optics":
        estimator = OPTICS(max_eps=max_eps, min_samples=min_sample, metric=metric)
    else:
        raise ValueError("method (%s) is unsupported" % method)

    estimator.fit(x)
    label, count = np.unique(estimator.labels_, return_counts=True)
    if len(count) == 1:
        raise ValueError("a single cluster was formed")
    elif not np.any(label == -1):
        raise ValueError("no outlier points")

    outlier_indicies = np.where(estimator.labels_ == -1)[0]
    inlier_indicies = np.where(estimator.labels_ != -1)[0]
    if outlier_indicies.shape[0] < _min_samples(inlier_indicies.shape[0], n_outliers):
        raise ValueError("not enough outliers")

    check_random_state(random_state).shuffle(outlier_indicies)
    return _make_outlier_arrays(
        x,
        n_outliers=_n_outliers(
            n_outliers, outlier_indicies.shape[0], inlier_indicies.shape[0]
        ),
        outlier_indicies=outlier_indicies,
        inlier_indicies=inlier_indicies,
    )


def majority_outliers(x, y, *, n_outliers=0.05, random_state=None):
    """Labels the majority class as inliers

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_timestep)
        The input samples

    y : ndarray of shape (n_samples, )
        The input labels.

    n_outliers : float, optional
        The number of outlier samples expressed as a fraction of the inlier samples.

    random_state : int or RandomState
        - If `int`, `random_state` is the seed used by the random number generator
        - If `RandomState` instance, `random_state` is the random number generator
        - If `None`, the random number generator is the `RandomState` instance used
            by `np.random`.

    Returns
    -------
    x_outlier : ndarray of shape (n_inliers + n_outliers, n_timestep)
        The samples

    y_outlier : ndarray of shape (n_inliers + n_outliers, )
        The inliers (labeled as 1) and outlier (labled as -1)
    """
    x = check_array(x, allow_multivariate=True)
    y = check_array(y, ensure_2d=False)
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) < 2:
        raise ValueError("require more than 1 labels, got %r" % len(labels))

    outlier_label_ = labels[labels != labels[np.argmax(counts)]]
    random_state = check_random_state(random_state)

    outlier_indicator = np.isin(y, outlier_label_)
    outlier_indicies = outlier_indicator.nonzero()[0]
    inlier_indicies = (~outlier_indicator).nonzero()[0]
    random_state.shuffle(outlier_indicies)

    return _make_outlier_arrays(
        x,
        n_outliers=_n_outliers(
            n_outliers, outlier_indicies.shape[0], inlier_indicies.shape[0]
        ),
        outlier_indicies=outlier_indicies,
        inlier_indicies=inlier_indicies,
    )


def minority_outliers(x, y, *, n_outliers=0.05, random_state=None):
    """Labels (a fraction of) the minority class as the outlier.

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_timestep)
        The input samples

    y : ndarray of shape (n_samples, )
        The input labels.

    n_outliers : float, optional
        The number of outlier samples expressed as a fraction of the inlier samples.

        - if float, the number of outliers are guaranteed but an error is raised
          if the minority class has to few samples.

    random_state : int or RandomState
        - If `int`, `random_state` is the seed used by the random number generator
        - If `RandomState` instance, `random_state` is the random number generator
        - If `None`, the random number generator is the `RandomState` instance used
            by `np.random`.

    Returns
    -------
    x_outlier : ndarray of shape (n_inliers + n_outliers, n_timestep)
        The samples

    y_outlier : ndarray of shape (n_inliers + n_outliers, )
        The inliers (labeled as 1) and outlier (labled as -1)
    """
    x = check_array(x, allow_multivariate=True)
    y = check_array(y, ensure_2d=False)

    labels, label_count = np.unique(y, return_counts=True)
    if len(labels) < 2:
        raise ValueError("require more than 1 labels, got %r" % len(labels))

    min_label = np.argmin(label_count)

    min_samples = _min_samples(x.shape[0] - label_count[min_label], n_outliers)
    if label_count[min_label] < min_samples:
        raise ValueError("not enough samples of the minority class")

    outlier_label_ = labels[min_label]
    random_state = check_random_state(random_state)

    outlier_indicies = np.where(y == outlier_label_)[0]
    random_state.shuffle(outlier_indicies)
    inlier_indicies = np.where(y != outlier_label_)[0]

    return _make_outlier_arrays(
        x,
        n_outliers=_n_outliers(
            n_outliers, outlier_indicies.shape[0], inlier_indicies.shape[0]
        ),
        outlier_indicies=outlier_indicies,
        inlier_indicies=inlier_indicies,
    )


def emmott_outliers(
    x,
    y,
    *,
    n_outliers=None,
    confusion_estimator=None,
    difficulty_estimator=None,
    difficulty="simplest",
    scale=None,
    variation="tight",
    random_state=None,
):
    """Create a synthetic outlier detection dataset from a labeled classification
    dataset using the method described by Emmott et.al. (2013).

    The Emmott labeler can reliably label both binary and multiclass datasets. For
    binary datasets a random label is selected as the outlier class. For multiclass
    datasets a set of classes with maximal confusion (as measured by
    ``confusion_estimator`` is selected as outlier label. For each outlier sample the
    ``difficulty_estimator`` assigns a difficulty score which is digitized into ranges
    and selected according to the ``difficulty`` parameters. Finally a sample of
    approximately ``n_outlier`` is selected either maximally dispersed or tight.

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_timestep)
        The input samples

    y : ndarray of shape (n_samples, )
        The input labels.

    n_outliers : float, optional
        The number of outlier samples expressed as a fraction of the inlier samples.

        - if float, the number of outliers are guaranteed but an error is raised
          if the the requested difficulty has to few samples or the labels selected
          for the outlier label has to few samples.

    confusion_estimator : object, optional
        Estimator of class confusion for datasets where ``n_classes > 2``. Default to a
        random forest classifier.

    difficulty_estimator : object, optional
        Estimator for sample difficulty. The difficulty estimator must support
        ``predict_proba``. Defaults to a kernel logistic regression model with
        a RBF-kernel.

    difficulty : {'any', 'simplest', 'hardest'}, int or array-like, optional
        The difficulty of the outlier points quantized according to scale. The value
        should be in the range ``[1, len(scale)]`` with lower difficulty denoting
        simpler outliers. If an array is given, multiple difficulties can be
        included, e.g., ``[1, 4]`` would mix easy and difficult outliers.

        - if 'any' outliers are sampled from all scores
        - if 'simplest' the simplest n_outliers are selected
        - if 'hardest' the hardest n_outliers are selected

    scale : array-like, optional
        The scale of quantized difficulty scores. Defaults to ``[0, 0.16, 0.3, 0.5]``.
        Scores (which are probabilities in the range [0, 1]) are fit into the ranges
        using ``np.digitize(difficulty, scale)``.

    variation : {'tight', 'dispersed'}, optional
        Selection procedure for sampling outlier samples

        - if 'tight' a pivot point is selected and the ``n_outlier`` closest samples
          are selected according to their euclidean distance
        - if 'dispersed' ``n_outlier`` points are selected according to a facility
          location algorithm such that they are distributed among the outliers.

    random_state : int or RandomState
        - If `int`, `random_state` is the seed used by the random number generator
        - If `RandomState` instance, `random_state` is the random number generator
        - If `None`, the random number generator is the `RandomState` instance used
            by `np.random`.

    Returns
    -------
    x_outlier : ndarray of shape (n_inliers + n_outliers, n_timestep)
        The samples

    y_outlier : ndarray of shape (n_inliers + n_outliers, )
        The inliers (labeled as 1) and outlier (labled as -1)

    Notes
    -----
    - For multiclass datasets the Emmott labeler require the package `networkx`
    - For dispersed outlier selection the Emmott labeler require the package
      `scikit-learn-extra`

    The difficulty parameters 'simplest' and 'hardest' are not described by
    Emmott et.al. (2013)

    Warnings
    --------
    n_outliers
        The number of outliers returned is dependent on the difficulty setting and the
        available number of samples of the minority class. If the minority class does
        not contain sufficient number of samples of the desired difficulty, fewer than
        n_outliers may be returned.

    References
    ----------
    Emmott, A. F., Das, S., Dietterich, T., Fern, A., & Wong, W. K. (2013).
        Systematic construction of anomaly detection benchmarks from real data.
        In Proceedings of the ACM SIGKDD workshop on outlier detection and description
        (pp. 16-21).

    """
    x = check_array(x, dtype=float)
    y = check_array(y, ensure_1d=True)
    random_state = check_random_state(random_state)
    n_classes = np.unique(y).shape[0]

    if n_classes > 2:
        outlier_label = _emmott_multiclass_outlier_class(
            x, y, confusion_estimator, random_state, n_classes
        )
    elif n_classes == 2:
        labels, counts = np.unique(y, return_counts=True)
        outlier_label = labels[np.argmin(counts)]
    else:
        raise ValueError("require more than 1 labels, got %r" % n_classes)

    outlier_indicator = np.isin(y, outlier_label)
    if np.sum(outlier_indicator) < _min_samples(
        x.shape[0] - outlier_indicator.shape[0], n_outliers
    ):
        raise ValueError("not enough samples of the minority class")

    y_new = np.ones(x.shape[0], dtype=int)
    y_new[outlier_indicator] = -1

    if difficulty_estimator is None:
        difficulty_estimator = KernelLogisticRegression(
            kernel="poly", max_iter=1000, random_state=random_state
        )
    else:
        difficulty_estimator = clone(difficulty_estimator)
        _set_random_states(difficulty_estimator, random_state)

    outliers_indices = np.where(y_new == -1)[0]
    inlier_indices = np.where(y_new == 1)[0]
    x_outliers = x[outliers_indices]
    y_outliers = y_new[outliers_indices]

    difficulty_estimate = _emmott_estimate_difficulty(x, y_new, difficulty_estimator)
    difficulty_estimate = difficulty_estimate[outliers_indices].reshape(-1)

    n_outliers_ = _n_outliers(n_outliers, y_outliers.shape[0], inlier_indices.shape[0])

    if n_outliers_ < y_outliers.shape[0]:
        scale = scale if scale is not None else _DEFAULT_EMMOTT_SCALE
        difficulty_scores = np.digitize(difficulty_estimate, scale)
        if isinstance(difficulty, str):
            if difficulty == "any":
                outlier_selector = np.arange(0, difficulty_estimate.shape[0])
            elif difficulty == "simplest":
                outlier_selector = np.argpartition(difficulty_estimate, n_outliers_)[
                    :n_outliers_
                ]
            elif difficulty == "hardest":
                outlier_selector = np.argpartition(difficulty_estimate, -n_outliers_)[
                    -n_outliers_:
                ]
            else:
                raise ValueError("difficulty (%s) is not supported" % difficulty)
        else:
            outlier_selector = np.isin(difficulty_scores, difficulty)
            min_samples = _min_samples(inlier_indices.shape[0], n_outliers)
            if np.sum(outlier_selector) < min_samples:
                scores, counts = np.unique(difficulty_scores, return_counts=True)
                raise ValueError(
                    "not enough samples (%d) with the requested "
                    "difficulty %s, available %s"
                    % (
                        min_samples,
                        difficulty,
                        ", ".join(["%d: %d" % (s, c) for s, c in zip(scores, counts)]),
                    )
                )
        x_outliers = x_outliers[outlier_selector]

    if variation in _EMMOTT_VARIATION:
        variation = _EMMOTT_VARIATION[variation]
    else:
        raise ValueError("variation (%s) is not supported" % variation)

    outlier_sampled = variation(
        x_outliers, n_outliers_, random_state.randint(np.iinfo(np.int32).max)
    )

    return (
        np.concatenate([x[inlier_indices], x_outliers[outlier_sampled]], axis=0),
        np.concatenate([y_new[inlier_indices], y_outliers[outlier_sampled]], axis=0),
    )


def _make_outlier_arrays(x, *, n_outliers, outlier_indicies, inlier_indicies):
    x_outlier = x[outlier_indicies[:n_outliers]]
    x_inlier = x[inlier_indicies]
    x = np.concatenate([x_outlier, x_inlier], axis=0)
    y = np.ones(x.shape[0])
    y[: x_outlier.shape[0]] = -1
    return x, y


def _min_samples(n_samples, n_outliers):
    if n_outliers is None:
        min_sample = 1
    else:
        if not 0 < n_outliers <= 1.0:
            raise ValueError(
                "n_outliers must be in the range ]0, 1], got %f" % n_outliers
            )
        min_sample = n_outliers * n_samples
    return min_sample


def _n_outliers(n_outliers, total_outliers, n_inliers):
    if n_outliers is None:
        return total_outliers
    elif isinstance(n_outliers, numbers.Real):
        if not 0.0 < n_outliers <= 1.0:
            raise ValueError("n_outliers must be in (0, 1], got %r" % n_outliers)
        return min(total_outliers, math.ceil(n_outliers * n_inliers))
    else:
        raise ValueError("n_outlier (%r) is not supported" % n_outliers)


def _emmott_estimate_difficulty(
    x,
    y_new,
    difficulty_estimator,
):
    difficulty_estimator.fit(x, y_new)
    if hasattr(difficulty_estimator, "oob_decision_function_"):
        difficulty_estimate = difficulty_estimator.oob_decision_function_
    else:
        difficulty_estimate = difficulty_estimator.predict_proba(x)

    difficulty_estimate = difficulty_estimate[
        :, np.where(difficulty_estimator.classes_ == 1)[0]
    ]

    return difficulty_estimate


def _emmott_multiclass_outlier_class(
    x, y, confusion_estimator, random_state, n_classes
):
    if confusion_estimator is None:
        confusion_estimator = RandomForestClassifier(n_jobs=-1, oob_score=True)
    else:
        confusion_estimator = clone(confusion_estimator)
    _set_random_states(confusion_estimator, random_state)

    confusion_estimator.fit(x, y)
    try:
        import networkx as nx

        if hasattr(confusion_estimator, "oob_decision_function_"):
            y_pred = confusion_estimator.classes_[
                np.argmax(confusion_estimator.oob_decision_function_, axis=1)
            ]
        else:
            y_pred = confusion_estimator.predict(x)
        cm = confusion_matrix(y, y_pred)  # TODO: use probabilities
        graph = nx.Graph()
        classes = confusion_estimator.classes_
        graph.add_nodes_from(classes)
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j:
                    graph.add_edge(classes[i], classes[j], weight=cm[i][j] + cm[j][i])

        max_spanning_tree = nx.maximum_spanning_tree(graph, algorithm="kruskal")
        coloring = nx.algorithms.bipartite.color(max_spanning_tree)
        labeling = {1: [], 0: []}
        for cls in classes:
            labeling[coloring[cls]].append(cls)

        zero = np.isin(y, labeling[0])
        one = np.isin(y, labeling[1])

        if np.sum(zero) <= np.sum(one):
            outlier_label = np.array(labeling[0])
        else:
            outlier_label = np.array(labeling[1])
    except ImportError as e:
        _soft_dependency_error(e, package="networkx", context="emmott_outlier")

    return outlier_label


def _variation_dispersed(x, n_outliers, random_state):
    try:
        from sklearn_extra.cluster import KMedoids

        n_outliers = min(x.shape[0], n_outliers)
        f = KMedoids(n_clusters=n_outliers, random_state=random_state).fit(x)
        return f.medoid_indices_
    except ModuleNotFoundError as e:
        _soft_dependency_error(
            e, package="scikit-learn-extra", context="variation='dispersed'"
        )


def _variation_tight(x, n_outliers, random_state):
    random_state = check_random_state(random_state)
    n_outliers = min(x.shape[0], n_outliers)
    if n_outliers == x.shape[0]:
        return np.arange(0, x.shape[0])
    sample = random_state.randint(0, x.shape[0])
    f = NearestNeighbors(n_neighbors=n_outliers).fit(x)
    return f.kneighbors(x[sample].reshape(1, -1), return_distance=False).reshape(-1)


def _set_random_states(estimator, random_state=None):
    random_state = check_random_state(random_state)
    to_set = {}
    for key in sorted(estimator.get_params(deep=True)):
        if key == "random_state" or key.endswith("__random_state"):
            to_set[key] = random_state.randint(np.iinfo(np.int32).max)

    if to_set:
        estimator.set_params(**to_set)


_EMMOTT_VARIATION = {"tight": _variation_tight, "dispersed": _variation_dispersed}
_DEFAULT_EMMOTT_SCALE = np.array([0, 0.16, 0.3, 0.5])
