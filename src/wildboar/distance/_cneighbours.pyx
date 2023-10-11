# cython: language_level=3
# cython: infer_types=True
# cython: boundscheck=False
# Fast swap step and build step in PAM algorithm for k_medoid.
# Author: Timothée Mathieu
# License: 3-clause BSD
# Adapted from: https://github.com/scikit-learn-contrib/scikit-learn-extra
# /blob/main/sklearn_extra/cluster/_k_medoids_helper.pyx#L72

import numpy as np


def _pam_optimal_swap(
    double[:, :] D,
    Py_ssize_t[:] medoid_idxs,
    Py_ssize_t[:] not_medoid_idxs,
    double[:] Djs,
    double[:] Ejs,
    Py_ssize_t n_clusters
):
    # Initialize best cost change and the associated swap couple.
    cdef (Py_ssize_t, Py_ssize_t, double) best_cost_change = (1, 1, 0.0)
    cdef Py_ssize_t sample_size = D.shape[0]
    cdef Py_ssize_t i, j, h, id_i, id_h, id_j
    cdef double cost_change
    cdef Py_ssize_t not_medoid_shape = sample_size - n_clusters
    cdef bint cluster_i_bool, not_cluster_i_bool, second_best_medoid
    cdef bint not_second_best_medoid

    # Compute the change in cost for each swap.
    for h in range(not_medoid_shape):
        # id of the potential new medoid.
        id_h = not_medoid_idxs[h]
        for i in range(n_clusters):
            # id of the medoid we want to replace.
            id_i = medoid_idxs[i]
            cost_change = 0.0
            # compute for all not-selected points the change in cost
            for j in range(not_medoid_shape):
                id_j = not_medoid_idxs[j]
                cluster_i_bool = D[id_i, id_j] == Djs[id_j]
                not_cluster_i_bool = D[id_i, id_j] != Djs[id_j]
                second_best_medoid = D[id_h, id_j] < Ejs[id_j]
                not_second_best_medoid = D[id_h, id_j] >= Ejs[id_j]

                if cluster_i_bool & second_best_medoid:
                    cost_change += D[id_j, id_h] - Djs[id_j]
                elif cluster_i_bool & not_second_best_medoid:
                    cost_change += Ejs[id_j] - Djs[id_j]
                elif not_cluster_i_bool & (D[id_j, id_h] < Djs[id_j]):
                    cost_change += D[id_j, id_h] - Djs[id_j]

            # same for i
            second_best_medoid = D[id_h, id_i] < Ejs[id_i]
            if  second_best_medoid:
                cost_change += D[id_i, id_h]
            else:
                cost_change += Ejs[id_i]

            if cost_change < best_cost_change[2]:
                best_cost_change = (id_i, id_h, cost_change)

    # If one of the swap decrease the objective, return that swap.
    if best_cost_change[2] < 0:
        return best_cost_change
    else:
        return None


def _pam_build(double[:, :] D, Py_ssize_t n_clusters):
    """Compute BUILD initialization, a greedy medoid initialization."""

    cdef Py_ssize_t[:] medoid_idxs = np.zeros(n_clusters, dtype=int)
    cdef Py_ssize_t sample_size = len(D)
    cdef Py_ssize_t[:] not_medoid_idxs = np.arange(sample_size, dtype=int)
    cdef Py_ssize_t i, j,  id_i, id_j

    medoid_idxs[0] = np.argmin(np.sum(D, axis=0))
    not_medoid_idxs = np.delete(not_medoid_idxs, medoid_idxs[0])

    cdef Py_ssize_t n_medoids_current = 1

    cdef double[:] Dj = D[medoid_idxs[0]].copy()
    cdef double cost_change
    cdef (Py_ssize_t, Py_ssize_t) new_medoid = (0, 0)
    cdef double cost_change_max

    for _ in range(n_clusters -1):
        cost_change_max = 0

        for i in range(sample_size - n_medoids_current):
            id_i = not_medoid_idxs[i]
            cost_change = 0

            for j in range(sample_size - n_medoids_current):
                id_j = not_medoid_idxs[j]
                cost_change += max(0, Dj[id_j] - D[id_i, id_j])

            if cost_change >= cost_change_max:
                cost_change_max = cost_change
                new_medoid = (id_i, i)

        medoid_idxs[n_medoids_current] = new_medoid[0]
        n_medoids_current += 1
        not_medoid_idxs = np.delete(not_medoid_idxs, new_medoid[1])

        for id_j in range(sample_size):
            Dj[id_j] = min(Dj[id_j], D[id_j, new_medoid[0]])
    return medoid_idxs.base
