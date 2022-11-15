# Authors: Isak Samsten
# License: BSD 3 clause

import math
import numbers
import warnings

import numpy as np
from sklearn import clone
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from sklearn.utils.validation import _is_arraylike, check_is_fitted

from ..distance import pairwise_distance
from ..transform import IntervalTransform
from ..utils import _soft_dependency_error
from ..utils.validation import check_array, check_option, check_X_y

__all__ = [
    "KernelLogisticRegression",
    "kmeans_outliers",
    "density_outliers",
    "minority_outliers",
    "majority_outliers",
    "emmott_outliers",
]


class KernelLogisticRegression(LogisticRegression):
    """A simple kernel logistic implementation using a Nystroem kernel approximation

    See Also
    --------
    wildboar.datasets.outlier.EmmottLabeler : Synthetic outlier dataset construction

    """

    def __init__(
        self,
        kernel=None,
        *,
        kernel_params=None,
        n_components=100,
        penalty="l2",
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        multi_class="auto",
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None,
    ):
        """
        Parameters
        ----------
        kernel : str, optional
            The kernel function to use. See `sklearn.metrics.pairwise.kernel_metric`
            for kernels. The default kernel is 'rbf'.

        kernel_params : dict, optional
            Parameters to the kernel function.

        n_components : int, optional
            Number of features to construct
        """
        super().__init__(
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio,
        )
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.n_components = n_components

    def fit(self, x, y, sample_weight=None):
        random_state = check_random_state(self.random_state)
        kernel = self.kernel or "rbf"
        n_components = min(x.shape[0], self.n_components)
        self.nystroem_ = Nystroem(
            kernel=kernel,
            kernel_params=self.kernel_params,
            n_components=n_components,
            random_state=random_state.randint(np.iinfo(np.int32).max),
        )
        self.nystroem_.fit(x)
        super().fit(self.nystroem_.transform(x), y, sample_weight=sample_weight)
        return self

    def decision_function(self, x):
        check_is_fitted(self)
        return super().decision_function(self.nystroem_.transform(x))


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
    centroid_distance = pairwise_distance(k_means_.cluster_centers_, metric="euclidean")

    # skip self matches and invalid clusters
    centroid_distance[centroid_distance == 0] = np.nan

    if np.all(np.isnan(centroid_distance)):
        raise ValueError("There are no valid clusters.")

    # hide the warning for ignored clusters
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        outlier_cluster_ = np.nanargmax(np.nanmean(centroid_distance, axis=1))

    return _make_outlier_arrays(
        x,
        n_outliers=n_outliers,
        outlier_indicies=np.where(k_means_.labels_ == outlier_cluster_)[0],
        inlier_indicies=np.where(k_means_.labels_ != outlier_cluster_)[0],
        random_state=random_state,
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
    """Labels samples as outliers if a density cluster algorithm fail to assign them to
    a cluster

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
        raise ValueError("method must be 'dbscan' or 'optics', got %s." % method)

    estimator.fit(x)
    label, count = np.unique(estimator.labels_, return_counts=True)
    if len(count) == 1:
        raise ValueError("A single cluster was formed.")
    elif not np.any(label == -1):
        raise ValueError("There are no outlier points.")

    return _make_outlier_arrays(
        x,
        n_outliers=n_outliers,
        outlier_indicies=np.where(estimator.labels_ == -1)[0],
        inlier_indicies=np.where(estimator.labels_ != -1)[0],
        random_state=check_random_state(random_state),
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
    x = check_array(x, allow_3d=True)
    y = check_array(y, ensure_2d=False)
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) < 2:
        raise ValueError("Two labels are required, got %r" % len(labels))

    outlier_indicator = np.isin(y, labels[labels != labels[np.argmax(counts)]])
    return _make_outlier_arrays(
        x,
        n_outliers=n_outliers,
        outlier_indicies=outlier_indicator.nonzero()[0],
        inlier_indicies=(~outlier_indicator).nonzero()[0],
        random_state=check_random_state(random_state),
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
    x = check_array(x, allow_3d=True)
    y = check_array(y, ensure_2d=False)

    labels, label_count = np.unique(y, return_counts=True)
    if len(labels) < 2:
        raise ValueError("Two labels are required, got %r" % len(labels))

    outlier_label = labels[np.argmin(label_count)]
    return _make_outlier_arrays(
        x,
        n_outliers=n_outliers,
        outlier_indicies=np.where(y == outlier_label)[0],
        inlier_indicies=np.where(y != outlier_label)[0],
        random_state=check_random_state(random_state),
    )


def emmott_outliers(
    x,
    y,
    *,
    n_outliers=None,
    confusion_estimator=None,
    difficulty_estimator=None,
    transform="interval",
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

    transform : 'interval' or Transform, optional
        Transform x before the confusion and difficulty estimator.

        - if None, no transformation is applied.
        - if 'interval', use the :class:`transform.IntervalTransform` with default
          parameters.
        - otherwise, use the supplied transform

    difficulty : {'any', 'simplest', 'hardest'}, int or array-like, optional
        The difficulty of the outlier points quantized according to scale. The value
        should be in the range ``[1, len(scale)]`` with lower difficulty denoting
        simpler outliers. If an array is given, multiple difficulties can be
        included, e.g., ``[1, 4]`` would mix easy and difficult outliers.

        - if 'any' outliers are sampled from all scores
        - if 'simplest' the simplest n_outliers are selected
        - if 'hardest' the hardest n_outliers are selected

    scale : int or array-like, optional
        The scale of quantized difficulty scores. Defaults to ``[0, 0.16, 0.3, 0.5]``.
        Scores (which are probabilities in the range [0, 1]) are fit into the ranges
        using ``np.digitize(difficulty, scale)``.

        - if int, use `scale` percentiles based in the difficulty scores.

    variation : {'tight', 'dispersed'}, optional
        Selection procedure for sampling outlier samples. If ``difficulty="simplest"``
        or ``difficulty="hardest"``, this parameter has no effect.

        - if 'tight' a pivot point is selected and the ``n_outlier`` closest samples
          are selected according to their euclidean distance.
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
    x, y = check_X_y(x, y, dtype=float)
    random_state = check_random_state(random_state)
    n_classes = np.unique(y).shape[0]

    if transform == "interval":
        transform = IntervalTransform()
    elif transform is not None:
        transform = clone(transform)

    if transform is not None:
        transform.fit(x)

    if n_classes > 2:
        if confusion_estimator is None:
            confusion_estimator = RandomForestClassifier(n_jobs=-1, oob_score=True)
        else:
            confusion_estimator = clone(confusion_estimator)

        _set_random_states(
            confusion_estimator, random_state.randint(np.iinfo(np.int32).max)
        )
        outlier_label = _emmott_multiclass_outlier_class(
            x,
            y,
            confusion_estimator=confusion_estimator,
            transform=transform,
            n_labels=n_classes,
        )
    elif n_classes == 2:
        labels, counts = np.unique(y, return_counts=True)
        outlier_label = labels[np.argmin(counts)]
    else:
        raise ValueError("Two labels are required, got %r" % n_classes)

    outlier_indicator = np.isin(y, outlier_label)
    y_new = np.ones(x.shape[0], dtype=int)
    y_new[outlier_indicator] = -1

    if difficulty_estimator is None:
        difficulty_estimator = KernelLogisticRegression(
            kernel="poly",
            max_iter=1000,
            random_state=random_state.randint(np.iinfo(np.int32).max),
        )
    else:
        difficulty_estimator = clone(difficulty_estimator)
        _set_random_states(
            difficulty_estimator, random_state.randint(np.iinfo(np.int32).max)
        )

    outliers_indices = np.where(y_new == -1)[0]
    inlier_indices = np.where(y_new == 1)[0]
    x_outliers = x[outliers_indices]
    y_outliers = y_new[outliers_indices]

    difficulty_estimate = _emmott_estimate_difficulty(
        x, y_new, difficulty_estimator=difficulty_estimator, transform=transform
    )
    difficulty_estimate = difficulty_estimate[outliers_indices].reshape(-1)

    outlier_fraction = y_outliers.shape[0] / x.shape[0]
    if outlier_fraction > n_outliers:
        n_outliers_ = min(
            y_outliers.shape[0], math.ceil(n_outliers * inlier_indices.size)
        )

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
                raise ValueError(
                    "difficulty must be 'any', 'simplest' or 'hardest', got %r."
                    % difficulty
                )
        else:
            if scale is None:
                scale = _DEFAULT_EMMOTT_SCALE
            elif isinstance(scale, numbers.Integral):
                scale = np.percentile(
                    difficulty_estimate, np.linspace(0, 100, scale, endpoint=False)
                )
            elif _is_arraylike(scale):
                scale = np.array(scale)
            else:
                raise ValueError(
                    "scale must be int or array-like, not %r" % type(scale).__qualname__
                )
            difficulty_scores = np.digitize(difficulty_estimate, scale)
            difficulties = np.unique(difficulty_scores)
            if difficulty not in difficulties:
                options = ", ".join([str(d) for d in difficulties[:-1]])
                raise ValueError(
                    f"difficulty must be {options} or {difficulties[-1]}, "
                    f"got {difficulty}"
                )
            outlier_selector = np.isin(difficulty_scores, difficulty)

        x_outliers = x_outliers[outlier_selector]
        variation = check_option(_EMMOTT_VARIATION, variation, "variation")
        outlier_sampled = variation(
            x_outliers, n_outliers_, random_state.randint(np.iinfo(np.int32).max)
        )
    else:
        outlier_sampled = np.arange(y_outliers.shape[0])

    if outlier_sampled.shape[0] / x.shape[0] < n_outliers:
        n_inliers = math.ceil(
            outlier_sampled.shape[0] / n_outliers - outlier_sampled.shape[0]
        )
        random_state.shuffle(inlier_indices)
        inlier_indices = inlier_indices[:n_inliers]

    return (
        np.concatenate([x[inlier_indices], x_outliers[outlier_sampled]], axis=0),
        np.concatenate([y_new[inlier_indices], y_outliers[outlier_sampled]], axis=0),
    )


def _make_outlier_arrays(
    x, *, n_outliers, outlier_indicies, inlier_indicies, random_state
):
    random_state.shuffle(outlier_indicies)
    random_state.shuffle(inlier_indicies)
    outlier_fraction = outlier_indicies.size / x.shape[0]
    if outlier_fraction < n_outliers:
        n_inliers = math.ceil(
            outlier_indicies.size / n_outliers - outlier_indicies.size
        )
        n_outliers = outlier_indicies.size
    elif outlier_fraction > n_outliers:
        n_outliers = min(
            outlier_indicies.size, math.ceil(n_outliers * inlier_indicies.size)
        )
        n_inliers = inlier_indicies.size
    else:
        n_outliers = outlier_indicies.size
        n_inliers = inlier_indicies.size

    x_outlier = x[outlier_indicies[:n_outliers]]
    x_inlier = x[inlier_indicies[:n_inliers]]
    x = np.concatenate([x_outlier, x_inlier], axis=0)
    y = np.ones(x.shape[0])
    y[: x_outlier.shape[0]] = -1
    return x, y


def _emmott_estimate_difficulty(
    x,
    y_new,
    *,
    difficulty_estimator,
    transform=None,
):
    if transform is not None:
        x = transform.transform(x)

    difficulty_estimator.fit(x, y_new)
    if hasattr(difficulty_estimator, "oob_decision_function_"):
        difficulty_estimate = difficulty_estimator.oob_decision_function_
    else:
        difficulty_estimate = difficulty_estimator.predict_proba(x)

    difficulty_estimate = difficulty_estimate[
        :, np.where(difficulty_estimator.classes_ == 1)[0]
    ]

    return difficulty_estimate


def _emmott_multiclass_outlier_class(x, y, *, confusion_estimator, transform, n_labels):
    if transform is not None:
        transform.transform(x)

    confusion_estimator.fit(x, y)
    try:
        import networkx as nx

        if hasattr(confusion_estimator, "oob_decision_function_"):
            y_pred = confusion_estimator.classes_[
                np.argmax(confusion_estimator.oob_decision_function_, axis=1)
            ]
        else:
            y_pred = confusion_estimator.predict(transform.transform(x))

        cm = confusion_matrix(y, y_pred)  # TODO: use probabilities
        graph = nx.Graph()
        labels = confusion_estimator.classes_
        graph.add_nodes_from(labels)
        for i in range(n_labels):
            for j in range(n_labels):
                if i != j:
                    graph.add_edge(labels[i], labels[j], weight=cm[i][j] + cm[j][i])

        max_spanning_tree = nx.maximum_spanning_tree(graph, algorithm="kruskal")
        coloring = nx.algorithms.bipartite.color(max_spanning_tree)
        labeling = {1: [], 0: []}
        for label in labels:
            labeling[coloring[label]].append(label)

        zero = np.isin(y, labeling[0])
        one = np.isin(y, labeling[1])

        if np.sum(zero) <= np.sum(one):
            outlier_label = np.array(labeling[0])
        else:
            outlier_label = np.array(labeling[1])
    except ModuleNotFoundError as e:
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
