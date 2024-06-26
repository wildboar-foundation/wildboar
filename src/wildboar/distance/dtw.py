# Authors: Isak Samsten
# License: BSD 3 clause

"""
DTW alignment and distance computations.

The :mod:`wildboar.distance.dtw` module implements several functions for
computing DTW alignments and distances.
"""

import math
import numbers
from functools import partial

import numpy as np
from sklearn.utils import check_random_state, check_scalar
from sklearn.utils.validation import _check_sample_weight, _is_arraylike

from ..utils.validation import check_array
from ._distance import pairwise_distance
from ._elastic import _dtw_alignment, _dtw_envelop, _dtw_lb_keogh

__all__ = [
    "dtw_alignment",
    "wdtw_alignment",
    "dtw_distance",
    "wdtw_distance",
    "ddtw_distance",
    "wddtw_distance",
    "dtw_mapping",
    "dtw_envelop",
    "dtw_lb_keogh",
    "jeong_weight",
    "dtw_average",
]


def _compute_warp_size(x_size, r, *, y_size=0):
    check_scalar(r, "r", numbers.Real, min_val=0, max_val=1)
    return max(math.floor(max(x_size, y_size) * r), 1)


def dtw_distance(x, y, *, r=1.0):
    """
    Compute the dynamic time warping distance.

    Parameters
    ----------
    x : array-like of shape (x_timestep, )
        The first time series.
    y : array-like of shape (y_timestep, )
        The second time series.
    r : float, optional
        The warping window in [0, 1] as a fraction of `max(x_timestep, y_timestep)`.

    Returns
    -------
    float
        The dynamic time warping distance.

    See Also
    --------
    dtw_alignment : compute the dtw alignment matrix
    """
    x = check_array(x, ravel_1d=True, ensure_2d=False)
    y = check_array(y, ravel_1d=True, ensure_2d=False)
    return pairwise_distance(x, y, metric="dtw", metric_params={"r": r})


def ddtw_distance(x, y, *, r=1.0):
    """
    Compute the derivative dynamic time warping distance.

    Parameters
    ----------
    x : array-like of shape (x_timestep, )
        The first time series.
    y : array-like of shape (y_timestep, )
        The second time series.
    r : float, optional
        The warping window in [0, 1] as a fraction of max(x_timestep, y_timestep).

    Returns
    -------
    float
        The dynamic time warping distance.

    See Also
    --------
    dtw_distance : compute the dtw distance
    """
    x = check_array(x, ravel_1d=True, ensure_2d=False)
    y = check_array(y, ravel_1d=True, ensure_2d=False)
    return pairwise_distance(x, y, metric="ddtw", metric_params={"r": r})


def wdtw_distance(x, y, *, r=1.0, g=0.05):
    """
    Compute the weighted dynamic time warping distance.

    Parameters
    ----------
    x : array-like of shape (x_timestep, )
        The first time series.
    y : array-like of shape (y_timestep, )
        The second time series.
    r : float, optional
        The warping window in [0, 1] as a fraction of `max(x_timestep, y_timestep)`.
    g : float, optional
        Penalization for points deviating the diagonal.

    Returns
    -------
    float
        The dynamic time warping distance.

    See Also
    --------
    dtw_distance : compute the dtw distance
    """
    x = check_array(x, ravel_1d=True, ensure_2d=False)
    y = check_array(y, ravel_1d=True, ensure_2d=False)
    return pairwise_distance(x, y, metric="wdtw", metric_params={"r": r, "g": g})


def wddtw_distance(x, y, *, r=1.0, g=0.05):
    """
    Compute the weighted derivative dynamic time warping distance.

    Parameters
    ----------
    x : array-like of shape (x_timestep, )
        The first time series.
    y : array-like of shape (y_timestep, )
        The second time series.
    r : float, optional
        The warping window in [0, 1] as a fraction of `max(x_timestep, y_timestep)`.
    g : float, optional
        Penalization for points deviating the diagonal.

    Returns
    -------
    float
        The dynamic time warping distance.

    See Also
    --------
    dtw_distance : compute the dtw distance
    """
    x = check_array(x, ravel_1d=True, ensure_2d=False)
    y = check_array(y, ravel_1d=True, ensure_2d=False)
    return pairwise_distance(x, y, metric="wddtw", metric_params={"r": r, "g": g})


def dtw_envelop(x, *, r=1.0):
    """
    Compute the envelop for LB_keogh.

    Parameters
    ----------
    x : array-like of shape (x_timestep,)
        The time series.
    r : float, optional
        The warping window in [0, 1] as a fraction of `max(x_timestep, y_timestep)`.

    Returns
    -------
    lower : ndarray of shape (x_timestep,)
        The min value of the envelop.
    upper : ndarray of shape (x_timestep,)
        The max value of the envelop.

    References
    ----------
    Keogh, E. (2002).
        Exact indexing of dynamic time warping.
        In 28th International Conference on Very Large Data Bases.
    """
    x = check_array(x, ravel_1d=True, ensure_2d=False, dtype=float, input_name="x")
    warp_size = _compute_warp_size(x.shape[0], r)
    if warp_size == x.shape[0]:
        warp_size -= 1

    return _dtw_envelop(x, warp_size)


def dtw_lb_keogh(x, y=None, *, lower=None, upper=None, r=1.0):
    """
    LB_keogh lower bound.

    Parameters
    ----------
    x : array-like of shape (x_timestep,)
        The first time series.
    y : array-like of shape (x_timestep,), optional
        The second time series (same size as x).
    lower : ndarray of shape (x_timestep,), optional
        The min value of the envelop.
    upper : ndarray of shape (x_timestep,), optional
        The max value of the envelop.
    r : float, optional
        The warping window in [0, 1] as a fraction of `max(x_timestep, y_timestep)`.

    Returns
    -------
    min_dist : float
        The cumulative minimum distance.
    lb_keogh : ndarray of shape (x_timestep,),
        The lower bound at each time step.

    Notes
    -----
    - if y=None, both lower and upper must be given
    - if y is given, lower and upper are ignored
    - if lower and upper is given and y=None, r is ignored

    References
    ----------
    Keogh, E. (2002).
        Exact indexing of dynamic time warping.
        In 28th International Conference on Very Large Data Bases.
    """
    check_args = dict(ravel_1d=True, ensure_2d=False, dtype=float)
    x = check_array(x, **check_args, input_name="x")
    if y is not None:
        y = check_array(y, **check_args, input_name="y")
        if y.shape[0] != x.shape[0]:
            raise ValueError(
                "x (%d) and y (%d) must have the same number of timesteps"
                % (x.shape[0], y.shape[0])
            )
        lower, upper = dtw_envelop(y, r=r)
    elif lower is None or upper is None:
        raise ValueError("both y, lower and upper can't be None")

    lower = check_array(lower, **check_args, input_name="lower")
    upper = check_array(upper, **check_args, input_name="upper")
    if lower.shape[0] != upper.shape[0] or lower.shape[0] != x.shape[0]:
        raise ValueError(
            "lower (%d), upper (%d) and x (%d) have the same number of timesteps"
            % (lower.shape[0], upper.shape[0], x.shape[0])
        )
    return _dtw_lb_keogh(x, lower, upper)


def dtw_alignment(x, y, *, r=1.0, weight=None, out=None):
    """
    Compute the Dynamic time warping alignment matrix.

    Parameters
    ----------
    x : array-like of shape (x_timestep,)
        The first time series.
    y : array-like of shape (y_timestep,)
        The second time series.
    r : float, optional
        The warping window in [0, 1] as a fraction of max(x_timestep, y_timestep).
    weight : array-like of shape (max(x_timestep, y_timestep), ), optional
        A weighting vector to penalize warping.
    out : array-like of shape (x_timestep, y_timestep), optional
        Store the warping path in this array.

    Returns
    -------
    ndarray of shape (x_timestep, y_timestep)
        The dynamic time warping alignment matrix.

    See Also
    --------
    dtw_distance : compute the dtw distance

    Notes
    -----
    If only the distance between two series is required use `dtw_distance`
    instead.
    """
    x = check_array(x, ravel_1d=True, ensure_2d=False, dtype=float, input_name="x")
    y = check_array(y, ravel_1d=True, ensure_2d=False, dtype=float, input_name="y")
    warp_size = _compute_warp_size(x.shape[0], r, y_size=y.shape[0])
    if weight is not None:
        weight = check_array(
            weight, ravel_1d=True, ensure_2d=False, dtype=float, input_name="weight"
        )
        if weight.shape[0] != max(x.shape[0], y.shape[0]):
            raise ValueError(
                "weight must have the same size as max(x.size, y.size) %d, got %d"
                % (max(x.shape[0], y.shape[0]), weight.shape[0])
            )

    return _dtw_alignment(x, y, warp_size, weight, out)


def wdtw_alignment(x, y, *, r=1.0, g=0.5, out=None):
    """
    Weighted dynamic time warping alignment.

    Parameters
    ----------
    x : array-like of shape (x_timestep,)
        The first time series.
    y : array-like of shape (y_timestep,)
        The second time series.
    r : float, optional
        The warping window in [0, 1] as a fraction of `max(x_timestep, y_timestep)`.
    g : float, optional
        Weighting described by Jeong et al. (2011) using `g` as penalty
        control.

        ::

            w = 1.0 / 1.0 + np.exp(-g * (x - n_samples / 2))
    out : array-like of shape (x_timestep, y_timestep), optional
        Store the warping path in this array.

    Returns
    -------
    ndarray of shape (x_timestep, y_timestep)
        The dynamic time warping alignment matrix.

    References
    ----------
    Jeong, Y., Jeong, M., Omitaomu, O. (2021)
        Weighted dynamic time warping for time series classification.
        Pattern Recognition 44, 2231-2240
    """
    weight = jeong_weight(max(x.shape[0], y.shape[0]), g)
    return dtw_alignment(x, y, r=r, weight=weight, out=out)


def jeong_weight(n, g=0.05):
    """
    Weighting described by Jeong et. al. (2011).

    Uses `g` as the penalty control.

    ::

        w = 1.0 / 1.0 + np.exp(-g * (x - n_samples / 2))

    Parameters
    ----------
    n : int
        The number of weights.
    g : float, optional
        Penalty control.

    Returns
    -------
    ndarray of shape (n, )
        The weights.

    References
    ----------
    Jeong, Y., Jeong, M., Omitaomu, O. (2021)
        Weighted dynamic time warping for time series classification.
        Pattern Recognition 44, 2231-2240
    """
    weight = 1.0 / (1.0 + np.exp(-g * (np.arange(n, dtype=float) - n / 2.0)))
    return weight


def dtw_mapping(x=None, y=None, *, alignment=None, r=1, return_index=False):
    """
    Optimal warping path between two series or from a given alignment matrix.

    Parameters
    ----------
    x : array-like of shape (x_timestep,), optional
        The first time series.
    y : array-like of shape (y_timestep,), optional
        The second time series.
    alignment : ndarray of shape (x_timestep, y_timestep), optional
        Precomputed alignment.
    r : float, optional
        The warping window in [0, 1] as a fraction of `max(x_timestep, y_timestep)`.
    return_index : bool, optional
        Return the indices of the warping path.

    Returns
    -------
    indicator : ndarray of shape (x_timestep, y_timestep)
        Boolean array with the dtw path.
    (x_indices, y_indices) : tuple, optional
        The indices of the first and second dimension of
        the optimal alignment path..

    Notes
    -----
    - either x and y or alignment must be provided
    - if alignment is given x and y are ignored
    - if alignment is given r is ignored
    """
    if alignment is None:
        if x is None or y is None:
            raise ValueError("if alignment=None, neither x or y can be None")
        alignment = dtw_alignment(x, y, r=r)
    else:
        alignment = check_array(
            alignment,
            force_all_finite=False,
            order=None,
            allow_eos=True,  # FIXME!
            input_name="alignment",
        )

    indicator = np.zeros(alignment.shape, dtype=bool)
    i = alignment.shape[0] - 1
    j = alignment.shape[1] - 1
    while i > 0 or j > 0:
        indicator[i, j] = True
        option_diag = alignment[i - 1, j - 1] if i > 0 and j > 0 else np.inf
        option_up = alignment[i - 1, j] if i > 0 else np.inf
        option_left = alignment[i, j - 1] if j > 0 else np.inf
        move = np.argmin([option_diag, option_up, option_left])
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    indicator[0, 0] = True
    if return_index:
        return indicator, indicator.nonzero()
    else:
        return indicator


def dtw_average(
    X,
    *,
    r=1.0,
    g=None,
    sample_weight=None,
    init="random",
    method="mm",
    max_stable=5,
    learning_rate=0.1,
    decay=0.9,
    tol=1e-5,
    max_epoch=50,
    return_cost=False,
    verbose=False,
    random_state=None,
):
    """
    Compute the DTW barycenter average (DBA).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_timestep)
        The samples to average.
    r : float, optional
        The warping window as a fraction of n_timestep.
    g : float, optional
        If set, use the weighted DTW alignment with `g` as penalty control.

        ::

            w(x)= 1.0 / 1.0 + np.exp(-g * (x - m / 2)),
    sample_weight : array-like of shape (n_samples, ), optional
        The sample weight. This influences how much each sample
        contributes to the average.
    init : "random" or array-like of shape (m_timestep, ), optional
        The initial sample used for average.
    method : {"mm", "ssg"}, optional
        The method for computing the DBA.

        - if "mm", use the majorize-minimize mean algorithm [1], which is equivalent to
          the DBA method in [2].

        - if "ssg", use the stochastic subgradient mean algorithm [1].
    max_stable : int, optional
        The maximum number of epoch where the average with lowest cost is unchanged
        if `method='ssg'`.
    learning_rate : float, optional
        The learning rate, if `method="ssg"`.
    decay : float, optional
        The learning rate decay, if `method="ssg"`.
    tol : float, optional
        The minmum change in cost between two epochs, if `method="mm"`.
    max_epoch : int, optional
        The maximum number of epochs.
    return_cost : bool, optional
        Return the averaging cost if `True`.
    verbose : bool, optional
        If set, show runtime information.
    random_state : int or RandomState, optional
        Pseudo-random number generator.

    Returns
    -------
    mean : array-like of shape (m_timestep, ) or (n_timestep, )
        The mean time series.
    cost : float, optional
        Return the cost of the average, if `return_cost=True`.

    Examples
    --------
    >>> from wildboar.datasets import load_two_lead_ecg
    >>> from wildboar.distance.dtw import dtw_average
    >>> X, y = load_two_lead_ecg()
    >>> dtw_average(X[:5], method="mm", random_state=1)
    array([-2.27442791e-01,  3.19807473e-02,  1.77490053e-01,  1.60441308e-01,
            2.31930140e-01,  2.17437783e-01,  2.43925941e-01,  2.60983434e-01,
            2.72118437e-01,  7.73352049e-02, -1.56701557e-02, -5.53269314e-02,
           -7.33366128e-02, -1.09010828e-01, -1.97539989e-01, -1.71443248e-01,
           -1.71443248e-01, -1.71443248e-01, -2.42492836e-01, -1.71408958e-01,
           -1.71408958e-01, -1.71408958e-01, -1.71408958e-01, -1.71408958e-01,
           -1.71408958e-01, -1.71408958e-01, -1.82518334e-01, -3.35671953e-01,
            1.26442901e-01, -7.38342948e-02, -9.11248815e-01, -1.99355168e+00,
           -2.08588712e+00, -2.35954194e+00, -2.78345146e+00, -2.41023092e+00,
           -1.99915956e+00, -1.82717462e+00, -1.82717462e+00, -1.71687181e+00,
           -1.55819192e+00, -1.28805337e+00, -1.06653283e+00, -7.25159669e-01,
           -4.02389872e-01, -2.39410523e-01,  2.34687887e-03,  2.98654485e-01,
            4.85832342e-01,  6.56436416e-01,  7.25302660e-01,  7.77697444e-01,
            8.24606299e-01,  8.76357782e-01,  9.27083874e-01,  9.44590342e-01,
            9.44590342e-01,  9.44590342e-01,  9.44590342e-01,  9.44590342e-01,
            9.64184026e-01,  1.03608265e+00,  1.13964118e+00,  1.33595675e+00,
            1.09954847e+00,  9.61924171e-01,  9.61924171e-01,  9.61924171e-01,
            9.61924171e-01,  9.61924171e-01,  9.61924171e-01,  9.47433305e-01,
            8.29583168e-01,  7.00425122e-01,  5.80524683e-01,  4.70210329e-01,
            4.40259039e-01,  3.59657389e-01,  3.52170730e-01,  3.54666287e-01,
            1.93690730e-01,  2.23968406e-01])
    """
    X = check_array(X, ensure_min_samples=2)
    r = check_scalar(r, "r", numbers.Real, min_val=0.0, max_val=1.0)
    random_state = check_random_state(random_state)

    if isinstance(init, str) and init == "random":
        mean = X[random_state.randint(X.shape[0])].copy()
    elif _is_arraylike(init):
        mean = np.array(init, copy=True)
    else:
        raise ValueError(
            "init must be array-like or 'random', not %r" % type(init).__qualname__
        )

    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X)

    if g is None:
        distancefn = partial(pairwise_distance, metric="dtw", metric_params=dict(r=r))
        alignmentfn = partial(dtw_alignment, r=r)
    else:
        g = check_scalar(g, "g", numbers.Real, min_val=0, include_boundaries="neither")
        distancefn = partial(
            pairwise_distance, metric="wdtw", metric_params=dict(r=r, g=g)
        )
        alignmentfn = partial(wdtw_alignment, r=r, g=g)

    def costfn(mean, X):
        cost = distancefn(mean, X)
        if sample_weight is None:
            return np.mean(cost)
        else:
            return np.average(cost, weights=sample_weight)

    if method == "mm":
        mean, cost = _mm_dtw_average(
            mean,
            X,
            sample_weight=sample_weight,
            max_epoch=max_epoch,
            costfn=costfn,
            alignmentfn=alignmentfn,
            tol=tol,
            verbose=verbose,
        )
    elif method == "ssg":
        mean, cost = _ssg_dtw_average(
            mean,
            X,
            sample_weight=sample_weight,
            max_epoch=max_epoch,
            max_stable=max_stable,
            learning_rate=learning_rate,
            decay=decay,
            costfn=costfn,
            alignmentfn=alignmentfn,
            verbose=verbose,
            random_state=random_state,
        )
    else:
        raise ValueError("method must be 'mm' or 'ssg', got %r" % method)

    if return_cost:
        return mean, cost
    else:
        return mean


# Implementation based on Stochastic Subgradient mean algorithm by:
# David Schultz and Brijnesh J. Jain. (2016).
#   Sample Mean Algorithms for Averaging in Dynamic Time Warping Spaces.
#   Zenodo. https://doi.org/10.5281/zenodo.216233
def _ssg_dtw_average(
    mean,
    X,
    *,
    sample_weight,
    max_epoch,
    learning_rate,
    decay,
    max_stable,
    costfn,
    alignmentfn,
    verbose,
    random_state,
):
    best_mean = None
    order = np.arange(X.shape[0])
    min_cost = np.inf
    n_stable = 0

    z = np.empty(mean.shape[0], dtype=float)
    for epoch in range(max_epoch):
        if n_stable > max_stable:
            if verbose:
                print(f"Completed at epoch={epoch} with cost={min_cost}.")
            break

        random_state.shuffle(order)
        for i, o in enumerate(order):
            z.fill(0)
            align_m, align_x = dtw_mapping(alignment=alignmentfn(mean, X[o])).nonzero()

            w = 1.0
            if sample_weight is not None:
                w = sample_weight[o]

            for m, x in zip(align_m, align_x):
                z[m] += mean[m] - X[o, x] * w

            mean -= learning_rate * z

            if epoch == 0:
                learning_rate = decay**i * learning_rate

        cost = costfn(mean, X)
        if cost < min_cost:
            if verbose:
                print(
                    f"New min cost={cost} at epoch={epoch} with "
                    f"learning_rate={learning_rate}."
                )

            min_cost = cost
            n_stable = 0
            best_mean = mean.copy()
        else:
            n_stable += 1

    return best_mean, min_cost


# Implementation based on Majorize-Minimize mean algorithm by:
# David Schultz and Brijnesh J. Jain. (2016).
#   Sample Mean Algorithms for Averaging in Dynamic Time Warping Spaces.
#   Zenodo. https://doi.org/10.5281/zenodo.216233
def _mm_dtw_average(
    mean, X, *, sample_weight, max_epoch, costfn, alignmentfn, tol, verbose
):
    prev_cost = np.inf
    cost = costfn(mean, X)
    z = np.empty(mean.shape[0], dtype=float)
    V = np.empty(mean.shape[0], dtype=float)
    for epoch in range(max_epoch):
        z.fill(0)
        V.fill(0)
        for i in range(X.shape[0]):
            align_m, align_x = dtw_mapping(alignment=alignmentfn(mean, X[i])).nonzero()
            w = 1.0
            if sample_weight is not None:
                w = sample_weight[i]
            for m, x in zip(align_m, align_x):
                V[m] += w
                z[m] += X[i, x] * w

        mean = z / V

        prev_cost = cost
        cost = costfn(mean, X)

        if abs(prev_cost - cost) < tol:
            if verbose:
                print(f"Complete at epoch={epoch} with cost={cost}.")
            break

    return mean, cost
