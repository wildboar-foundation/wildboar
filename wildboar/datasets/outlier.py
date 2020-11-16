import warnings

import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


def kmeans_labeler(x, n_clusters=None, min_count=3, random_state=None):
    """Label a dataset with normal samples and anomalous samples.

    An outlier is defined as a point belonging to a cluster that has the largest average distance to the other cluster
    centroids.

    Parameters
    ----------
    x : array-like of shape `(n_samples, n_timestep)`
        The input data to label.

    n_clusters : int, optional
        The number of clusters to fit. Default: ``x.shape[0] // 4``.

    min_count : int, optional
        The minimum number of samples in a cluster.

    random_state : int or RandomState, optional
        The random state passed to k-means

    Returns
    -------
    y : ndarray
        Labels for samples in x, where 1 denotes normal samples and -1 denote anomalous samples
    """
    n_clusters = n_clusters or x.shape[0] // 4
    k_means = KMeans(n_clusters=n_clusters, random_state=random_state).fit(x)
    _, count = np.unique(k_means.labels_, return_counts=True)

    # skip clusters with to few samples
    invalid_clusters = np.where(count < min_count)

    centroid_distance = pairwise_distances(k_means.cluster_centers_, metric='euclidean')

    # skip self matches
    centroid_distance[centroid_distance == 0] = np.nan

    # skip invalid clusters
    centroid_distance[invalid_clusters] = np.nan
    if np.all(np.isnan(x)):
        raise ValueError("no valid clusters")

    # hide the warning for ignored clusters
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        outlier_cluster = np.nanargmax(np.nanmean(centroid_distance, axis=1))

    y = np.ones(x.shape[0])
    y[k_means.labels_ == outlier_cluster] = -1
    return y
