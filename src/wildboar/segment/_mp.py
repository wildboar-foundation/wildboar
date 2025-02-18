import math
import numbers

import numpy as np

from ..distance._cmatrix_profile import _paired_matrix_profile
from ..utils.validation import _check_ts_array
from ._base import BaseSegmenter


class FlussSegmenter(BaseSegmenter):
    """
    Segmenter using the MatrixProfile and corrected ARC curve.

    Compute the Fast Low-cost Unipotent Semantic Segmentation (FLUSS)
    as described by Gharghabi (2017).

    The algorithm works by analyzing similarity relationships in time series data:

    1. For each position in the time series:
       - It finds its nearest neighbor (most similar subsequence)
       - Creates an "arc" connecting these two positions

    2. The arc curve is computed by:
       - Counting how many arcs pass over each position (including all
         positions between the start and end points of each arc)
       - Normalizing the counts to account for edge effects

    3. The resulting curve is used to find segment boundaries:
       - Low points (valleys) in the arc curve indicate natural boundaries
       - These are positions with few similarity relationships crossing them
       - High arc counts suggest positions within coherent segments

    The intuition is that segment boundaries occur where the time series behavior
    changes, which is reflected by fewer similarity relationships (arcs) crossing
    these points.

    Parameters
    ----------
    n_segments : int, optional
        The number of segments.
    window : int or float, optional
        The window size.

        - if int, the exact window size.
        - if float, the window size expressed as a fraction of the time series
          length.
    exclude : int or float, optional
        The exclusion zone.

        - if float, expressed as a fraction of the window size.
        - if int, exact size.
    boundary : float, optional
        The boundary of the ignored region around each segment expressed as a
        fraction of the window size.
    metric : str or callable, optional
        The distance metric

        See ``_METRICS.keys()`` for a list of supported metrics.
    metric_params : dict, optional
        Parameters to the metric.

        Read more about the parameters in the
        :ref:`User guide <list_of_metrics>`.
    n_jobs : int, optional
        The number of parallel jobs to compute the matrix profile.

    Attributes
    ----------
    labels_ : list of shape (n_samples, )
        A list of n_samples lists with the start index of the segment.

    References
    ----------
    Gharghabi, Shaghayegh, et al. (2017)
        Matrix profile VIII: domain agnostic online semantic segmentation at superhuman
        performance levels. In proceedings of International Conference on Data Mining
    """

    def __init__(
        self,
        n_segments=1,
        *,
        window=1.0,
        exclude=0.2,
        boundary=0.1,
        metric="euclidean",
        metric_params=None,
        n_jobs=None,
    ):
        super().__init__(metric, metric_params)
        self.n_segments = n_segments
        self.window = window
        self.exclude = exclude
        self.boundary = boundary
        self.n_jobs = n_jobs

    # @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """
        Fit the segmenter.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps)
            The samples.
        y : ignored, optional
            Ignored.

        Returns
        -------
        self
            The estimator.
        """
        X = self._validate_data(X)
        if not isinstance(self.window, numbers.Integral):
            self.window_ = math.ceil(self.window * X.shape[-1])
        else:
            self.window_ = self.window

        if not isinstance(self.exclude, numbers.Integral):
            self.exclude_ = math.ceil(self.exclude * self.window_)
        else:
            self.exclude_ = self.exclude

        X_ts = _check_ts_array(X)
        _, mpi = _paired_matrix_profile(
            X_ts,
            X_ts,
            self.window_,
            0,
            self.exclude_,
            self.n_jobs,
        )
        self._fit_X = X
        self.labels_ = _segment(
            mpi=mpi,
            n_segments=self.n_segments,
            window=self.window_,
            exclude=self.exclude_,
            boundary=self.boundary,
        )
        return self


def _segment(
    *,
    mpi=None,
    n_segments,
    window,
    exclude,
    boundary,
):
    mpi = np.atleast_2d(mpi)

    boundary = math.ceil(window * boundary)
    segments = -np.ones((mpi.shape[0], n_segments), dtype=np.intp)

    index = np.arange(mpi.shape[-1])
    arc_curve_normalize = 2 * index * (mpi.shape[-1] - index) / mpi.shape[-1]
    arc_curve_normalize[0] = 10e-10
    for i in range(mpi.shape[0]):
        arc_curve = np.minimum(
            np.cumsum(
                np.bincount(np.minimum(index, mpi[i]), minlength=mpi.shape[-1])
                - np.bincount(np.maximum(index, mpi[i]), minlength=mpi.shape[-1])
            )
            / arc_curve_normalize,
            1,
        )
        arc_curve[:boundary] = np.inf
        arc_curve[-boundary:] = np.inf

        for j in range(n_segments):
            argmin = np.argmin(arc_curve)
            if arc_curve[argmin] == np.inf:
                break

            segments[i, j] = argmin
            start = max(segments[i, j] - boundary, 0)
            end = min(segments[i, j] + boundary, arc_curve.shape[0])
            arc_curve[start:end] = np.inf

    return list(_strip_missing_segments(segments))


def _strip_missing_segments(a):
    n, m = a.shape
    for i in range(n):
        first = np.where(a[i] == -1)[0]
        if first.size == 0:
            yield a[i, :m]
        else:
            yield a[i, : first[0]]
