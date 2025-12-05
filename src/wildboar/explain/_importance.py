# Authors: Isak Samsten
# License: BSD 3 clause

import math
import numbers
from abc import ABCMeta, abstractmethod
from collections import namedtuple

import numpy as np
from sklearn.base import _fit_context
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _check_multimetric_scoring, _MultimetricScorer
from sklearn.model_selection._validation import _aggregate_score_dicts
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, check_random_state, check_scalar

from ..base import BaseEstimator, ExplainerMixin
from ..distance import pairwise_subsequence_distance, subsequence_match
from ..transform._interval import IntervalTransform
from ..transform._sax import SAX
from ..transform._shapelet import RandomShapeletMixin, ShapeletTransform
from ..utils.validation import check_array, check_option

try:
    from matplotlib.cm import ScalarMappable
    from matplotlib.pyplot import get_cmap, subplots

    from ..utils.plot import MidpointNormalize, plot_frequency_domain, plot_time_domain
except ModuleNotFoundError as e:
    from ..utils import DependencyMissing

    matplotlib_missing = DependencyMissing(e, package="matplotlib")
    get_cmap = matplotlib_missing
    ScalarMappable = matplotlib_missing
    subplots = matplotlib_missing
    MidpointNormalize = matplotlib_missing
    plot_time_domain = matplotlib_missing

Importance = namedtuple("Importance", ["mean", "std", "full"])


def plot_importances(importances, ax=None, labels=None):
    """
    Plot the importances as a boxplot.

    Parameters
    ----------
    importances : Importance or dict
        The importances.
    ax : Axes, optional
        The axes to plot. If importances is dict, ax must contain at least
        len(importances) Axes objects.
    labels : array-like, optional
        The labels for the importances.

    Returns
    -------
    Axes
        The plotted Axes.
    """
    n_importances = len(importances) if isinstance(importances, dict) else 1
    if ax is None:
        _, ax = subplots(ncols=n_importances)

    if isinstance(importances, dict):
        for i, (title, importance) in enumerate(importances.items()):
            _plot_single_importance(importance, ax=ax[i], labels=labels)
            ax[i].set_title(title)
    else:
        _plot_single_importance(importances, ax=ax, labels=labels)

    return ax


def _plot_single_importance(importance, ax, labels):
    order = np.argsort(importance.mean)
    if labels is not None:
        labels = np.array(labels)

    bplot = ax.boxplot(
        importance.full[order].T,
        sym="",
        vert=False,
        patch_artist=True,
        labels=order if labels is None else labels[order],
        medianprops=dict(color="black"),
    )

    cmap = get_cmap("Dark2", importance.mean.size)
    for i, patch in enumerate(bplot["boxes"]):
        patch.set_facecolor(cmap(i))


def _intervals(n, n_intervals):
    for i in range(n_intervals):
        length = n // n_intervals
        start = i * length + min(i % n_intervals, n % n_intervals)
        if i % n_intervals < n % n_intervals:
            length += 1
        yield start, start + length


def _unpack_scores(orig_score, perm_score):
    importances = orig_score - perm_score
    return Importance(
        mean=np.mean(importances, axis=1),
        std=np.std(importances, axis=1),
        full=importances,
    )


class PermuteImportance(BaseEstimator, metaclass=ABCMeta):
    """
    Base class for permutation importance.

    Parameters
    ----------
    scoring : str, list, dict or callable, optional
        The scoring function. By default the estimators score function is used.
    n_repeat : int, optional
        The number of repeated permutations.
    random_state : int or RandomState, optional
        Controls the random resampling of the original dataset.

        - If `int`, `random_state` is the seed used by the random number generator
        - If `RandomState` instance, `random_state` is the random number generator
        - If `None`, the random number generator is the `RandomState` instance used
          by `np.random`.
    """

    _parameter_constraints: dict = {
        "scoring": [None, str, list, dict, callable],
        "n_repeat": [Interval(numbers.Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(self, *, scoring=None, n_repeat=1, random_state=None):
        self.scoring = scoring
        self.n_repeat = n_repeat
        self.random_state = random_state

    def _reset_component(self, X, component):
        """
        Reset the component.

        This method is called before each component is permuted. It can be used
        to precompute values that are needed for the permutation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps)
            The input data.
        component : object
            The component to permute.
        """
        pass

    @abstractmethod
    def _permute_component(self, X, component, random_state):
        """
        Permute the component.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps)
            The input data.
        component : object
            The component to permute.
        random_state : RandomState
            The random state.

        Returns
        -------
        array-like of shape (n_samples, n_timesteps)
            The permuted data.
        """
        pass

    def _fit(self, estimator, components, X, y, random_state, sample_weight=None):  # noqa: PLR0912
        """
        Fit the importance.

        Parameters
        ----------
        estimator : object
            The estimator.
        components : list
            The components to permute.
        X : array-like of shape (n_samples, n_timesteps)
            The input data.
        y : array-like of shape (n_samples, )
            The target values.
        random_state : RandomState
            The random state.
        sample_weight : array-like of shape (n_samples, ), optional
            The sample weights.
        """
        if callable(self.scoring):
            scoring = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scoring = check_scoring(estimator, self.scoring)
        else:
            scoring_dict = _check_multimetric_scoring(estimator, self.scoring)
            scoring = _MultimetricScorer(**scoring_dict)

        self.components_ = np.array(components, dtype=object)
        scores = []
        for component in self.components_:
            self._reset_component(X, component)

            rep_scores = []
            for rep in range(self.n_repeat):
                X_perm = self._permute_component(X, component, random_state)

                if sample_weight is not None:
                    score = scoring(estimator, X_perm, y, sample_weight=sample_weight)
                else:
                    score = scoring(estimator, X_perm, y)

                rep_scores.append(score)

            if isinstance(rep_scores[0], dict):
                scores.append(_aggregate_score_dicts(rep_scores))
            else:
                scores.append(rep_scores)

        if sample_weight is not None:
            self.baseline_score_ = scoring(estimator, X, y, sample_weight=sample_weight)
        else:
            self.baseline_score_ = scoring(estimator, X, y)

        if isinstance(self.baseline_score_, dict):
            self.importances_ = {
                name: _unpack_scores(
                    self.baseline_score_[name],
                    np.array([scores[i][name] for i in range(len(self.components_))]),
                )
                for name in self.baseline_score_
            }
        else:
            self.importances_ = _unpack_scores(self.baseline_score_, np.array(scores))


class IntervalImportance(ExplainerMixin, PermuteImportance):
    """
    Interval importance for time series.

    Parameters
    ----------
    scoring : str, list, dict or callable, optional
        The scoring function. By default the estimators score function is used.
    n_repeat : int, optional
        The number of repeated permutations.
    n_intervals : str, optional
        The number of intervals.

        - if "sqrt", the number of intervals is the square root of n_timestep.
        - if "log2", the number of intervals is the log2 of n_timestep.
        - if int, exact number of intervals.
    window : int, optional
        The window size. If specified, n_intervals is ignored and the number of
        intervals is computed such that each interval is (at least) of size window.
    depth : int, optional
        The depth of the dyadic intervals.
    coverage_probability : float, optional
        The probability that a time step is covered by an interval.
    variability : float, optional
        Controls the variability of the interval sizes.
    random_state : int or RandomState
        - If `int`, `random_state` is the seed used by the random number generator
        - If `RandomState` instance, `random_state` is the random number generator
        - If `None`, the random number generator is the `RandomState` instance used
            by `np.random`.

    Attributes
    ----------
    importances_ : dict or Importance
        The importance scores for each interval. If dict, one value per scoring
        function.

    components_ : ndarray of shape (n_intervals, 2)
        The interval start and end positions.

    """

    _parameter_constraints: dict = {
        **PermuteImportance._parameter_constraints,
        "n_intervals": [
            StrOptions({"sqrt", "log2", "log"}, deprecated={"log"}),
            Interval(numbers.Integral, 1, None, closed="left"),
        ],
        "window": [Interval(numbers.Integral, 1, None, closed="left"), None],
        "depth": [Interval(numbers.Integral, 1, None, closed="left"), None],
        "coverage_probability": [Interval(numbers.Real, 0, 1, closed="right"), None],
        "variability": [Interval(numbers.Real, 0, None, closed="neither"), None],
    }

    def __init__(
        self,
        *,
        scoring=None,
        n_repeat=5,
        n_intervals="sqrt",
        window=None,
        depth=None,
        coverage_probability=None,
        variability=1,
        random_state=None,
    ):
        super().__init__(scoring=scoring, n_repeat=n_repeat, random_state=random_state)
        self.n_intervals = n_intervals
        self.window = window
        self.depth = depth
        self.coverage_probability = coverage_probability
        self.variability = variability

    def _permute_component(self, X, component, random_state):
        X_perm = X.copy()
        start, end = component
        random_state.shuffle(X_perm[:, start:end])
        return X_perm

    def fit(self, estimator, X, y, sample_weight=None):
        self._validate_params()
        estimator = self._validate_estimator(estimator)
        X, y = self._validate_data(X, y, reset=False, allow_3d=False)
        random_state = check_random_state(self.random_state)

        intervals = "fixed"
        n_intervals = self.n_intervals
        extra_params = {}
        if self.depth is not None:
            intervals = "dyadic"
            extra_params = {"depth": self.depth}
        elif self.coverage_probability is not None:
            intervals = "random"
            extra_params = {
                "coverage_probability": self.coverage_probability,
                "variability": self.variability,
            }
        elif self.window is not None:
            if self.window > self.n_timesteps_in_:
                raise ValueError(
                    f"The window parameter of {type(self).__name__} must be "
                    "<= n_timesteps_in_"
                )
            n_intervals = self.n_timesteps_in_ // self.window

        interval_transform = IntervalTransform(
            intervals=intervals,
            n_intervals=n_intervals,
            random_state=random_state.randint(np.iinfo(np.int32).max),
            **extra_params,
        ).fit(X)

        components = [
            (start, start + length)
            for _dim, (start, length, _) in interval_transform.embedding_.attributes
        ]

        self._fit(
            estimator,
            components,
            X,
            y,
            random_state,
            sample_weight=sample_weight,
        )
        return self

    def explain(self, x, y=None):
        check_is_fitted(self)
        x = self._validate_data(x, reset=False)
        importances = np.zeros(self.n_timesteps_in_, dtype=float)
        counts = np.zeros(self.n_timesteps_in_, dtype=float)
        for i, (start, end) in enumerate(self.components_):
            importances[start:end] += self.importances_.mean[i] / (end - start)
            counts[start:end] += 1

        mask = counts > 0
        importances[mask] /= counts[mask]
        return np.broadcast_to(importances, (x.shape[0], self.n_timesteps_in_))

    def plot(  # noqa: PLR0912
        self,
        x=None,
        y=None,
        *,
        ax=None,
        scoring=None,
        k=None,
        n_samples=100,
        show_grid=True,
    ):
        check_is_fitted(self)
        if isinstance(self.importances_, dict):
            importances = check_option(self.importances_, scoring, "scoring")
        else:
            if scoring is not None and scoring != self.scoring:
                raise ValueError("scoring must be %s, got %r" % (self.scoring, scoring))
            importances = self.importances_

        if x is None:
            return plot_importances(
                importances,
                ax=ax,
                labels=["(%d, %d)" % (start, end) for start, end in self.components_],
            )

        importances = importances.mean
        if k is None:
            k = importances.shape[0]
        elif isinstance(k, numbers.Integral):
            k = check_scalar(
                k,
                "top_k",
                numbers.Integral,
                min_val=1,
                max_val=importances.shape[0],
            )
        elif isinstance(k, numbers.Real):
            k = check_scalar(
                k,
                "top_k",
                numbers.Real,
                min_val=0,
                max_val=1,
                include_boundaries="right",
            )
            k = math.ceil(k * importances.shape[0])
        else:
            raise TypeError(
                "top_k must be int or float, not %s." % type(k).__qualname__
            )

        if ax is None:
            fig, ax = subplots()
        else:
            fig = None

        order = np.argsort(importances)[: -(k + 1) : -1]
        norm = MidpointNormalize(
            # vmin=importances[order[-1]],
            # vmax=importances[order[0]],
            # midpoint=0.0,
            vmin=-1.0,
            vmax=1.0,
            midpoint=0.0,
        )
        cmap = get_cmap("jet")

        if show_grid:
            for start, _ in self.components_:
                ax.axvline(
                    start,
                    0,
                    1,
                    color="gray",
                    linestyle="dashed",
                    linewidth=0.5,
                    dashes=(5, 5),
                    zorder=-2,
                )

        for o in order:
            start, end = self.components_[o]
            ax.axvspan(
                start,
                end,
                0,
                1,
                alpha=0.2,
                color=cmap(norm(importances[o])),
                zorder=1,
            )

        xticks = [start for start, _ in self.components_]
        xticks.append(self.components_[-1][1])
        ax.set_xticks(xticks)
        ax.tick_params(axis="x", labelrotation=-70)

        plot_time_domain(x, y=y, n_samples=n_samples, ax=ax)
        mappable = ScalarMappable(cmap=cmap, norm=norm)
        if fig is not None:
            fig.colorbar(mappable, ax=ax)
            return ax
        else:
            return ax, mappable


def _exponential_frequency_bands(n_timesteps, n_bands, growth_factor):
    """
    Yield frequency bands for permutation.

    Returns frequency bands with configurable growth, ensuring consecutive
    indices within each band. Each band represents a range of FFT bins,
    where bin k corresponds to frequency f[k] = k * fs/N Hz, where:

    - fs is the sampling frequency
    - N is the number of samples (n_timesteps)
    - k is the bin index


    Parameter
    ---------
    n_timesteps : int
        The number of time steps.
    n_bands : int
        The number of frequency bands to create
    growth_factor : float
        The growth_factor parameter controls band sizes:

        - growth_factor > 1: exponential growth (more detail at lower frequencies)
        - growth_factor = 1: linear growth (equal-sized bands)
        - growth_factor < 1: logarithmic decay (more detail at higher frequencies)

    Returns
    -------
    list of tuples
        Tuples with start and end band indices.
    """
    n_frequencies = n_timesteps // 2
    if n_timesteps % 2 == 0:  # even length, exclude Nyquist bin
        n_frequencies -= 1
    if n_bands > n_frequencies:
        raise ValueError(
            f"The window parameter of {type(self).__name__} must be <= n_timesteps // 2"
        )

    if growth_factor == 1:
        bands = np.linspace(1, n_frequencies, n_bands + 1, dtype=int)
    else:
        bands = np.zeros(n_bands + 1, dtype=int)
        bands[0] = 1  # skip DC component
        for i in range(1, n_bands + 1):
            bands[i] = int(
                max(
                    bands[i - 1] + 1,  # Ensure at least 1 bin difference
                    np.round(
                        1
                        + (n_frequencies - 1)
                        * ((growth_factor**i - 1) / (growth_factor**n_bands - 1))
                    ),
                )
            )
        bands[-1] = n_frequencies

    intervals = []
    for start, end in zip(bands[:-1], bands[1:]):
        intervals.append((start, end))

    return intervals


class FrequencyImportance(ExplainerMixin, PermuteImportance):
    """
    Explainer to evaluate feature importance based on frequency bands.

    This class implements a frequency-based importance measure by permuting values
    within frequency bands obtained from the Fourier transform of time series data.
    It extends PermuteImportance to analyze the impact of different frequency
    components on model predictions.

    Parameters
    ----------
    scoring : str, callable, or None, optional
        Scoring metric to evaluate importance. If None, uses estimator's score method.
    n_repeat : int, optional
        Number of times to permute each frequency band.
    random_state : int, RandomState instance or None, optional
        Controls randomization for permutations.
    n_bands : int, optional
        Number of frequency bands to analyze.
    spectrum : {"amplitude", "phase"}, optional
        Whether to permute amplitude components or to permute phase components.
    growth_factor : float, optional
        Controls growth of frequency band sizes:
        - growth_factor > 1: exponential growth (more detail at lower frequencies)
        - growth_factor = 1: linear growth (equal-sized bands)
        - growth_factor < 1: logarithmic decay (more detail at higher frequencies)

        Defaults to `np.e`.

    Attributes
    ----------
    components_ : list of tuples
        The frequency bands used for permutation, as (start, end) indices.
    importances_ : ImportanceContainer
        Contains the calculated feature importance scores.
    n_timesteps_in_ : int
        Number of timesteps in the input data.

    Notes
    -----
    The frequency bands are constructed by dividing the frequency domain into windows,
    with the size of each window controlled by the growth_factor parameter. This allows
    for analyzing different scales of temporal patterns in the data.

    See Also
    --------
    PermuteImportance : Base class for permutation-based importance.
    """

    _parameter_constraints: dict = {
        **PermuteImportance._parameter_constraints,
        "n_bands": [Interval(numbers.Integral, 1, None, closed="left")],
        "spectrum": [StrOptions({"phase", "amplitude"})],
        "growth_factor": [Interval(numbers.Real, 1, None, closed="left")],
    }

    def __init__(
        self,
        scoring=None,
        n_repeat=5,
        random_state=None,
        n_bands=10,
        spectrum="amplitude",
        growth_factor=np.e,
    ):
        super().__init__(scoring=scoring, n_repeat=n_repeat, random_state=random_state)
        self.n_bands = n_bands
        self.spectrum = spectrum
        self.growth_factor = growth_factor

    def _permute_component(self, X, component, random_state):
        start, end = component
        X_fft = np.fft.rfft(X, axis=1)

        idx = random_state.permutation(len(X_fft))

        amplitude = np.abs(X_fft)
        phase = np.angle(X_fft)

        if self.spectrum == "amplitude":
            amplitude[:, start:end] = amplitude[idx, start:end]
        else:
            phase[:, start:end] = phase[idx, start:end]

        X_fft_new = amplitude * np.exp(1j * phase)
        return np.fft.irfft(X_fft_new, n=X.shape[-1], axis=1)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, estimator, X, y, sample_weight=None):
        estimator = self._validate_estimator(estimator)
        X, y = self._validate_data(X, y, reset=False, allow_3d=False)
        random_state = check_random_state(self.random_state)
        self._fit(
            estimator,
            _exponential_frequency_bands(
                self.n_timesteps_in_, self.n_bands, self.growth_factor
            ),
            X,
            y,
            random_state,
            sample_weight=sample_weight,
        )
        return self

    def explain(self, X, y=None):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)
        X_fft = np.fft.rfft(X, axis=1)
        importances = np.zeros(X_fft.shape[1], dtype=float)
        counts = np.zeros(X_fft.shape[1], dtype=float)
        for i, (start, end) in enumerate(self.components_):
            importances[start:end] += self.importances_.mean[i] / (end - start)
            counts[start:end] += 1

        mask = counts > 0
        importances[mask] /= counts[mask]
        return np.broadcast_to(importances, (X.shape[0], X_fft.shape[1]))

    def plot(
        self,
        X=None,
        y=None,
        ax=None,
        k=None,
        sample_spacing=1,
        jitter=False,
        show_grid=True,
    ):
        check_is_fitted(self)
        frequencies = np.fft.fftfreq(self.n_timesteps_in_, d=sample_spacing)

        importances = self.importances_.mean
        if X is None:
            return plot_importances(
                self.importances_,
                ax=ax,
                labels=[
                    f"{frequencies[start]:.2f}Hz-{frequencies[end - 1]:.2f}Hz"
                    for start, end in self.components_
                ],
            )

        if ax is None:
            fig, ax = subplots()
        else:
            fig = None

        if show_grid:
            for start, _ in self.components_:
                ax.axvline(
                    start,
                    0,
                    1,
                    color="gray",
                    linestyle="dashed",
                    linewidth=0.5,
                    dashes=(5, 5),
                    zorder=-2,
                )

        order = np.argsort(importances)[::-1]
        norm = MidpointNormalize(
            vmin=-1,
            vmax=1,
            midpoint=0.0,
        )
        cmap = get_cmap("coolwarm")

        k = min(k, order.shape[0]) if k is not None else order.shape[0]
        for o in order[:k]:
            start, end = self.components_[o]
            ax.axvspan(
                start - 0.5,
                end - 0.5,
                0,
                1,
                alpha=0.2,
                color=cmap(norm(importances[o])),
                zorder=1,
            )

        plot_frequency_domain(
            X, y, bins=self.components_, ax=ax, spectrum=self.spectrum, jitter=jitter
        )
        ax.set_xticks(
            [
                (start + end) / 2 if end - start > 1 else start
                for start, end in self.components_
            ]
        )
        ax.set_xticklabels(
            [
                f"{frequencies[int((start + end) / 2) if end - start > 1 else start]:.2f}Hz"
                for start, end in self.components_
            ],
            rotation=90,
            ha="center",
        )
        ax.set
        mappable = ScalarMappable(cmap=cmap, norm=norm)
        if fig is not None:
            fig.colorbar(mappable, ax=ax)
            return ax
        else:
            return ax, mappable


class AmplitudeImportance(ExplainerMixin, PermuteImportance):
    """
    Compute the importance of equi-probable amplitude intervals.

    The implementation uses :class:`transform.SAX` to discretize the time
    series and then for each bin permute the samples along that bin.

    Parameters
    ----------
    scoring : str, list, dict or callable, optional
        The scoring function. By default the estimators score function is used.
    n_intervals : str or int, optional
        The number of intervals.

        - if "sqrt", the number of intervals is the square root of n_timestep.
        - if "log2", the number of intervals is the log2 of n_timestep.
        - if int, exact number of intervals.
    window : int, optional
        The window size. If specified, n_intervals is ignored and the number of
        intervals is computed such that each interval is (at least) of size window.
    binning : {"normal", "uniform"}, optional
        The binning strategy.

        - "normal": bins are computed such that they are equi-probable under a
          standard normal distribution.
        - "uniform": bins are computed such that they are equi-width.
    n_bins : int, optional
        The number of bins.
    n_repeat : int, optional
        The number of repeated permutations.
    random_state : int or RandomState, optional
        Controls the random resampling of the original dataset.

        - If `int`, `random_state` is the seed used by the random number generator
        - If `RandomState` instance, `random_state` is the random number generator
        - If `None`, the random number generator is the `RandomState` instance used
          by `np.random`.

    Attributes
    ----------
    importances_ : dict or Importance
        The importance scores for each interval. If dict, one value per scoring
        function.
    sax_ : SAX
        The fitted SAX transformer.
    """

    _parameter_constraints: dict = {
        **IntervalImportance._parameter_constraints,
        "binning": [StrOptions({"normal", "uniform"})],
        "n_bins": [Interval(numbers.Integral, 1, None, closed="left")],
    }
    _parameter_constraints.pop("depth")
    _parameter_constraints.pop("coverage_probability")
    _parameter_constraints.pop("variability")

    def __init__(
        self,
        scoring=None,
        n_intervals="sqrt",
        window=None,
        binning="normal",
        n_bins=4,
        n_repeat=1,
        random_state=None,
    ):
        super().__init__(
            scoring=scoring,
            n_repeat=n_repeat,
            random_state=random_state,
        )
        self.n_intervals = n_intervals
        self.window = window
        self.binning = binning
        self.n_bins = n_bins

    def _permute_component(self, X, component, random_state):
        X_perm = np.empty_like(X)
        idx = np.arange(X.shape[0])
        random_state.shuffle(idx)
        for i in range(X.shape[0]):
            j = idx[i]
            xi = self.x_sax_[i, :]
            xj = self.x_sax_[j, :]

            i_indicies = np.where(xi == component)[0]
            j_indicies = np.where(xj == component)[0]

            if i_indicies.size < j_indicies.size:
                j_indicies = random_state.choice(
                    j_indicies, i_indicies.size, replace=False
                )
            elif i_indicies.size > j_indicies.size:
                i_indicies = random_state.choice(
                    i_indicies, j_indicies.size, replace=False
                )

            X_perm[i] = X[i]
            for i_idx, j_idx in zip(i_indicies, j_indicies):
                i_start, i_end = self.sax_.intervals[i_idx]
                j_start, j_end = self.sax_.intervals[j_idx]

                if i_end - i_start < j_end - j_start:
                    j_end -= 1
                elif j_end - j_start < i_end - i_start:
                    i_end -= 1

                X_perm[i, i_start:i_end] = X[j, j_start:j_end]

        return X_perm

    def fit(self, estimator, X, y, sample_weight=None):
        self._validate_params()
        estimator = self._validate_estimator(estimator)
        X, y = self._validate_data(X, y, reset=False, allow_3d=False)
        self.sax_ = SAX(
            n_intervals=self.n_intervals,
            window=self.window,
            binning=self.binning,
            n_bins=self.n_bins,
            scale=True,
        ).fit(X)
        self.x_sax_ = self.sax_.transform(X)
        random_state = check_random_state(self.random_state)
        components = [b for b in range(self.n_bins)]
        self._fit(
            estimator,
            components,
            X,
            y,
            random_state,
            sample_weight=sample_weight,
        )
        return self

    def explain(self, X, y=None):
        X = self._validate_data(X, reset=False)
        X_sax = self.sax_.transform(X)
        return self.importances_.mean[X_sax]

    def _validate_data_plot(self, x):
        if (
            self.binning == "uniform"
            and not np.allclose(np.max(x, axis=1), 1)
            and not np.allclose(np.min(x, axis=1), 0)
        ):
            raise ValueError(
                "The SAX binning has been estimated from data, and the data to "
                "plot is not min/max scaled. Set preprocess=True to scale the data."
            )

        if (
            self.binning == "normal"
            and not np.allclose(np.mean(x, axis=1), 0)
            and not np.allclose(np.std(x, axis=1), 1)
        ):
            raise ValueError(
                "The SAX binning has been estimated from data, and the data to "
                "plot is not standardized. Set preprocess=True to scale the data."
            )

    def plot(  # noqa: PLR0915, PLR0912
        self,
        x=None,
        y=None,
        *,
        ax=None,
        n_samples=100,
        scoring=None,
        preprocess=True,
        k=None,
        show_bins=False,
        show_grid=True,
    ):
        """
        Plot the importances.

        If x is given, the importances are plotted over the samples optionally
        labeling each sample using the supplied labels. If x is not give, the
        importances are plotted as one or more boxplots.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_timesteps), optional
            The samples.
        y : array-like of shape (n_samples, ), optional
            The labels.
        ax : Axes, optional
            Axes to plot. If ax is set, x is None and scoring is None, the number of
            axes must be the same as the number of scorers.
        n_samples : int or float, optional
            The number of samples to plot, set to `None` to plot all.
        scoring : str, optional
            The scoring to plot if multiple scorers were used when fitting.
        preprocess : bool, optional
            Preprocess the time series to align with the bins, ignored if x is not None.
        k : int or float, optional
            The number of top bins to plot, ignored if x is not None.

            - if int, the specified number of bins are shown
            - if float, a fraction of the number of bins are shown.
        show_bins : bool, optional
            Annotate the plot with the index of the bin, ignored if x is not None.
        show_grid : bool, optional
            Annotate the plot with the bin thresholds, ignored if x is not None.

        Returns
        -------
        ax : Axis
            The axis.
        mappable : ScalarMappable, optional
            Return the mappable used to plot the colorbar.
            Only returned if ax is not None and x is not None.
        """
        check_is_fitted(self)
        if x is None:
            if scoring is not None and isinstance(self.importances_, dict):
                importances = check_option(self.importances_, scoring, "scoring")
            else:
                importances = self.importances_

            return plot_importances(importances, ax=ax)

        if isinstance(self.importances_, dict):
            importances = check_option(self.importances_, scoring, "scoring")
        else:
            if scoring is not None and scoring != self.scoring:
                raise ValueError(
                    "scoring must be '%s', got %r" % (self.scoring, scoring)
                )
            importances = self.importances_

        x = check_array(x)
        if preprocess:
            x = self.sax_.binning_.scale(x)
        else:
            self._validate_data_plot(x)

        importances = importances.mean
        if k is None:
            k = importances.shape[0]
        elif isinstance(k, numbers.Integral):
            k = check_scalar(
                k,
                "top_k",
                numbers.Integral,
                min_val=1,
                max_val=importances.shape[0],
            )
        elif isinstance(k, numbers.Real):
            k = check_scalar(
                k,
                "top_k",
                numbers.Real,
                min_val=0,
                max_val=1,
                include_boundaries="right",
            )
            k = math.ceil(k * importances.shape[0])
        else:
            raise TypeError(
                "top_k must be int or float, not %s." % type(k).__qualname__
            )

        if ax is None:
            fig, ax = subplots()
        else:
            fig = None

        order = np.argsort(importances)[: -(k + 1) : -1]
        norm = MidpointNormalize(
            # vmin=importances[order[-1]], vmax=importances[order[0]], midpoint=0.0
            vmin=-1,
            vmax=1,
            midpoint=0.0,
        )
        thresholds = self.sax_.binning_.get_thresholds()
        cmap = get_cmap("coolwarm")
        bottom, top = np.min(x), np.max(x)

        if show_grid:
            for threshold in thresholds:
                ax.axhline(
                    threshold,
                    0,
                    1,
                    color="gray",
                    linestyle="dashed",
                    linewidth=0.5,
                    dashes=(5, 5),
                    zorder=-2,
                )

        for i in order:
            if i == 0:
                start = bottom
                end = thresholds[i]
            elif i == importances.shape[0] - 1:
                start = thresholds[i - 1]
                end = top
            else:
                start = thresholds[i - 1]
                end = thresholds[i]

            span = ax.axhspan(
                start,
                end,
                0,
                1,
                alpha=0.3,
                zorder=1,
                color=cmap(norm(importances[i])),
                label="%i" % i,
            )

            if show_bins:
                ax.annotate("%d" % i, xy=(0, 0.5), va="center", xycoords=span)

        plot_time_domain(x, y, ax=ax, n_samples=n_samples)
        ax.set_ylim([np.min(x) - np.std(x), np.max(x) + np.std(x)])
        mappable = ScalarMappable(cmap=cmap, norm=norm)
        if fig is not None:
            fig.colorbar(mappable, ax=ax)
            return ax
        else:
            return ax, mappable


class ShapeletImportance(ExplainerMixin, PermuteImportance):
    """
    Compute the importance of shapelets.

    The importance is given by permuting time series sections with
    the minimum distance to shapelets.

    Parameters
    ----------
    scoring : str, list, dict or callable, optional
        The scoring function. By default the estimators score function is used.
    n_repeat : int, optional
        The number of repeated permutations.
    n_shapelets : int, optional
        The number of shapelets to sample for the explanation.
    k_best : int, optional
        Select the top-k shapelets according to `score_func`
    score_func : callable, optional
        Score function to evaluate the performance of a shapelet.
    alpha : float, optional
        Define matching shapelets as dist < min_dist * (1 + alpha). If None
        (default) only use the best match.
    min_shapelet_size : float, optional
        The minimum size of shapelets used for explanation.
    max_shapelet_size : float, optional
        The maximum size of shapelets used for explanation.
    coverage_probability : float, optional
        The probability that a time step is covered by a
        shapelet, in the range 0 < coverage_probability <= 1.

        - For larger `coverage_probability`, we get larger shapelets.
        - For smaller `coverage_probability`, we get shorter shapelets.
    variability : float, optional
        Controls the shape of the Beta distribution used to
        sample shapelets. Defaults to 1.

        - Higher `variability` creates more uniform intervals.
        - Lower `variability` creates more variable intervals sizes.
    metric : str, optional
        The metric.
    metric_params : str, optional
        The metric parameters.
    random_state : int or RandomState, optional
        Controls the random resampling of the original dataset.

        - If `int`, `random_state` is the seed used by the
          random number generator.
        - If :class:`numpy.random.RandomState` instance, `random_state` is the
          random number generator.
        - If `None`, the random number generator is the
          :class:`numpy.random.RandomState` instance used by
          :func:`numpy.random`.

    Attributes
    ----------
    components : ndarray
        The shapelets

    """

    _parameter_constraints: dict = {
        **PermuteImportance._parameter_constraints,
        **RandomShapeletMixin._parameter_constraints,
        "k_best": [Interval(numbers.Integral, 1, None, closed="neither"), None],
        "score_func": [callable, None],
        "alpha": [Interval(numbers.Real, 0.0, None, closed="neither"), None],
    }

    def __init__(
        self,
        scoring=None,
        n_repeat=1,
        n_shapelets=1000,
        k_best=None,
        alpha=None,
        score_func=None,
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        coverage_probability=None,
        variability=1,
        metric="euclidean",
        metric_params=None,
        random_state=None,
    ):
        super().__init__(
            scoring=scoring,
            n_repeat=n_repeat,
            random_state=random_state,
        )
        self.k_best = k_best
        self.score_func = score_func
        self.alpha = alpha
        self.n_shapelets = n_shapelets
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.coverage_probability = coverage_probability
        self.variability = variability
        self.metric = metric
        self.metric_params = metric_params

    def _reset_component(self, X, component):
        min_dist, self._shapelet_idx = pairwise_subsequence_distance(
            component,
            X,
            metric=self.metric,
            metric_params=self.metric_params,
            return_index=True,
        )
        if self.alpha is not None:
            self._threshold = min_dist * (1 + self.alpha)
            self._shapelet_idx = subsequence_match(
                component,
                X,
                metric=self.metric,
                metric_params=self.metric_params,
                threshold=self._threshold,
            )
        else:
            self._shapelet_idx = [np.array([idx]) for idx in self._shapelet_idx]

    def _permute_component(self, X, component, random_state):
        X_perm = np.empty_like(X)
        idx = np.arange(X.shape[0])
        random_state.shuffle(idx)
        for i in range(X.shape[0]):
            j = idx[i]

            i_shapelet_indices = self._shapelet_idx[i]
            j_shapelet_indices = self._shapelet_idx[j]

            X_perm[j] = X[j]
            if i != j and len(i_shapelet_indices) > 0:
                for j_shapelet_idx in j_shapelet_indices:
                    i_shapelet_idx = i_shapelet_indices[
                        random_state.randint(0, len(i_shapelet_indices))
                    ]
                    X_perm[j, j_shapelet_idx : (j_shapelet_idx + component.size)] = X[
                        i, i_shapelet_idx : (i_shapelet_idx + component.size)
                    ]

        return X_perm

    def fit(self, estimator, X, y, sample_weight=None):
        self._validate_params()
        self._validate_estimator(estimator)
        X, y = self._validate_data(X, y, reset=False)

        random_state = check_random_state(self.random_state)
        shapelet_transform = ShapeletTransform(
            n_shapelets=self.n_shapelets,
            strategy="random",
            metric=self.metric,
            min_shapelet_size=self.min_shapelet_size,
            max_shapelet_size=self.max_shapelet_size,
            coverage_probability=self.coverage_probability,
            variability=self.variability,
            random_state=random_state.randint(np.iinfo(np.int32).max),
        ).fit(X)
        if self.k_best is not None and self.k_best < self.n_shapelets:
            Xt = shapelet_transform.transform(X)
            score = self.score_func(Xt, y)
            k_best = np.argpartition(score, -self.k_best)[-self.k_best :]
        else:
            k_best = range(self.n_shapelets)

        attributes = shapelet_transform.embedding_.attributes
        components = [attributes[i][1][1] for i in k_best]
        self._fit(
            estimator,
            components,
            X,
            y,
            random_state,
            sample_weight=sample_weight,
        )
        return self

    def explain(self, X, y=None, kernel_scale=None):
        check_is_fitted(self)
        distances, indices = pairwise_subsequence_distance(
            self.components_,
            X,
            metric=self.metric,
            metric_params=self.metric_params,
            return_index=True,
        )
        if kernel_scale is not None:
            weights = self._distance_weight(distances, kernel_scale)
        else:
            weights = np.broadcast_to(1, shape=(X.shape[0], len(self.components_)))

        importances = self.importances_.mean
        explanation = np.zeros_like(X)
        norm = np.zeros_like(X)
        for j, shapelet in enumerate(self.components_):
            importance = importances[j]
            for i in range(X.shape[0]):
                index = indices[i, j]
                weight = weights[i, j]
                explanation[i, index : (index + shapelet.size)] += weight * importance
                norm[i, index : (index + shapelet.size)] += 1
        explanation[norm > 0] /= norm[norm > 0]
        return explanation

    def explain_shapelet(self, X, y=None, kernel_scale=0.25):
        check_is_fitted(self)
        distances, indices = pairwise_subsequence_distance(
            self.components_,
            X,
            metric=self.metric,
            metric_params=self.metric_params,
            return_index=True,
        )

        if kernel_scale is not None:
            weights = self._distance_weight(distances, kernel_scale)
        else:
            weights = np.broadcast_to(1, shape=(X.shape[0], len(self.components_)))
        importances = self.importances_.mean

        if y is None:
            explanation = np.zeros((len(self.components_), X.shape[-1]), dtype=float)
            for j, shapelet in enumerate(self.components_):
                importance = importances[j]
                for i in range(X.shape[0]):
                    index = indices[i, j]
                    weight = weights[i, j]
                    explanation[j, index : (index + shapelet.size)] += (
                        weight * importance
                    )

            return explanation / X.shape[0]
        else:
            labels, inv, counts = np.unique(y, return_inverse=True, return_counts=True)
            explanation = np.zeros(
                (len(self.components_), len(labels), X.shape[-1]), dtype=float
            )
            for j, shapelet in enumerate(self.components_):
                importance = importances[j]
                for i in range(X.shape[0]):
                    index = indices[i, j]
                    weight = weights[i, j]
                    explanation[j, inv[i], index : (index + shapelet.size)] += (
                        1 / counts[inv[i]] * weight * importance
                    )

            return explanation

    def plot(  # noqa: PLR0912
        self, X=None, y=None, k=None, scoring=None, kernel_scale=0.25, ax=None
    ):
        if X is None:
            return plot_importances(
                self.importances_, ax=ax, labels=range(len(self.components_))
            )

        if isinstance(self.importances_, dict):
            importances = check_option(self.importances_, scoring, "scoring")
        else:
            if scoring is not None and scoring != self.scoring:
                raise ValueError(
                    "scoring must be '%s', got %r" % (self.scoring, scoring)
                )
            importances = self.importances_

        importances = importances.mean
        order = np.argsort(importances)[::-1]

        if k is None:
            k = importances.size

        if y is None:
            explanation = self.explain_shapelet(X, kernel_scale=kernel_scale)
            if ax is None:
                fig, ax = subplots()
            else:
                fig = None

            mappable = ax.pcolormesh(explanation[order[:k]], cmap="coolwarm")
            ax.set_yticks(np.arange(k) + 0.5, order[:k])
            if fig is not None:
                fig.colorbar(mappable, ax=ax)
                return ax
            else:
                return ax, mappable
        else:
            explanation = self.explain_shapelet(X, y, kernel_scale=kernel_scale)
            distances, index = pairwise_subsequence_distance(
                self.components_,
                X,
                metric=self.metric,
                metric_params=self.metric_params,
                return_index=True,
            )
            cmap = get_cmap("coolwarm")
            norm = MidpointNormalize(
                vmin=-1.0,
                vmax=1.0,
                midpoint=0.0,
            )
            labels, inv, lbl_count = np.unique(
                y, return_inverse=True, return_counts=True
            )
            if ax is None:
                fig, ax = subplots(
                    nrows=k,
                    ncols=len(labels),
                    figsize=(len(labels) * 4, k * 2.5),
                    sharex=True,
                    sharey=True,
                )

            weights = (
                self._distance_weight(distances, kernel_scale)
                if kernel_scale is not None
                else np.broadcast_to(1, shape=(X.shape[0], len(self.components_)))
            )
            label_cmap = get_cmap("Dark2", lut=len(labels))
            for i in range(len(labels)):
                for j in range(k):
                    plot_time_domain(
                        X[y == labels[i]],
                        n_samples=lbl_count[i],
                        ax=ax[j, i],
                        color=label_cmap(i),
                    )

            for i in range(0, X.shape[0]):
                for j in range(k):
                    order_j = order[j]
                    ax[j, inv[i]].plot(
                        np.arange(
                            index[i, order_j],
                            index[i, order_j] + self.components_[order_j].size,
                        ),
                        self.components_[order_j],
                        linewidth=weights[i, order_j],
                        color=cmap(norm(importances[order_j])),
                    )

            for i in range(len(labels)):
                ax[0, i].set_title("Label=%r" % labels[i])

            for i in range(k):
                ax[i, 0].set_ylabel(order[i])

            mappable = ScalarMappable(norm=norm, cmap=cmap)
            if fig is not None:
                fig.colorbar(mappable, orientation="horizontal", ax=ax[-1, :])
                return ax
            else:
                return ax, mappable

    def _distance_weight(self, distances, kernel_scale=0.25):
        """
        Compute similarity weights using a Gaussian kernel.

        The weight of each sample is computed using a Gaussian (RBF) kernel, where the
        kernel width is adapted based on the size of each component. Samples with smaller
        distances get higher weights (closer to 1) while samples further away get lower
        weights (closer to 0).

        Parameters
        ----------
        distances : array-like
            The pairwise distances between samples
        kernel_scale : float, default=0.25
            Scaling factor for the kernel width. Higher values result in slower
            decay of weights with distance.

        Returns
        -------
        ndarray
            Computed weights for each sample, where each weight is in [0, 1]
        """
        kernel_width = [np.sqrt(s.size) * kernel_scale for s in self.components_]
        return np.sqrt(np.exp(-(distances**2) / np.array(kernel_width) ** 2))
