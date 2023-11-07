from numbers import Integral, Real

from sklearn.manifold import MDS as Sklearn_MDS  # noqa: N811
from sklearn.utils._param_validation import Interval, StrOptions

from ..base import BaseEstimator
from ..distance._distance import _METRICS, pairwise_distance


class MDS(BaseEstimator):
    """
    Multidimensional scaling.

    Parameters
    ----------
    n_components : int, optional
        Number of dimensions in which to immerse the dissimilarities.
    metric : bool, optional
        If `True`, perform metric MDS; otherwise, perform nonmetric MDS.
        When `False` (i.e. non-metric MDS), dissimilarities with 0 are considered as
        missing values.
    n_init : int, optional
        Number of times the SMACOF algorithm will be run with different
        initializations. The final results will be the best output of the runs,
        determined by the run with the smallest final stress.
    max_iter : int, optional
        Maximum number of iterations of the SMACOF algorithm for a single run.
    verbose : int, optional
        Level of verbosity.
    eps : float, optional
        Relative tolerance with respect to stress at which to declare
        convergence. The value of `eps` should be tuned separately depending
        on whether or not `normalized_stress` is being used.
    n_jobs : int, optional
        The number of jobs to use for the computation. If multiple
        initializations are used (``n_init``), each run of the algorithm is
        computed in parallel.
    random_state : int, RandomState instance or None, optional
        Determines the random number generator used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
    dissimilarity : str, optional
        The dissimilarity measure.

        See `_METRICS.keys()` for a list of supported metrics.
    dissimilarity_params : dict, optional
        Parameters to the dissimilarity measue.

        Read more about the parameters in the
        :ref:`User guide <list_of_metrics>`.
    normalized_stress : bool or "auto", optional
        Whether use and return normed stress value (Stress-1) instead of raw
        stress calculated by default. Only supported in non-metric MDS.

    Notes
    -----
    This implementation is a convenience wrapper around
    :class:`sklearn.manifold.MDS` to when using Wildboar metrics.
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "metric": ["boolean"],
        "n_init": [Interval(Integral, 1, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "verbose": ["verbose"],
        "eps": [Interval(Real, 0.0, None, closed="left")],
        "n_jobs": [None, Integral],
        "random_state": ["random_state"],
        "dissimilarity": [StrOptions(_METRICS.keys())],
        "dissimilarity_params": [None, dict],
        "normalized_stress": [
            "boolean",
            StrOptions({"auto"}),
        ],
    }

    def __init__(
        self,
        n_components=2,
        *,
        metric=True,
        n_init=4,
        max_iter=300,
        verbose=0,
        eps=1e-3,
        n_jobs=None,
        random_state=None,
        dissimilarity="euclidean",
        dissimilarity_params=None,
        normalized_stress="warn",
    ):
        self.n_components = n_components
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.verbose = verbose
        self.eps = eps
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.normalized_stress = normalized_stress
        self.dissimilarity = dissimilarity
        self.dissimilarity_params = dissimilarity_params

    def fit(self, x, y=None):
        x = self._validate_data(x, allow_3d=True)
        self.fit_transform(x)
        return self

    def fit_transform(self, x, y=None):
        self._validate_params()
        x = self._validate_data(x, allow_3d=True, reset=False)
        self.mds_ = Sklearn_MDS(
            n_components=self.n_components,
            metric=self.metric,
            n_init=self.n_init,
            max_iter=self.max_iter,
            verbose=self.verbose,
            eps=self.eps,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            dissimilarity="precomputed",
            normalized_stress=self.normalized_stress,
        )
        return self.mds_.fit_transform(
            pairwise_distance(
                x,
                dim="mean",
                metric=self.dissimilarity,
                n_jobs=self.n_jobs,
                metric_params=self.dissimilarity_params,
            )
        )
