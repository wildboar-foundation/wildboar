# Authors: Isak Samsten
# License: BSD 3 clause

import math
import numbers
from abc import ABCMeta, abstractmethod
from collections import namedtuple

import numpy as np
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _check_multimetric_scoring, _MultimetricScorer
from sklearn.model_selection._validation import _aggregate_score_dicts
from sklearn.utils.validation import check_random_state, check_scalar

from ..base import BaseEstimator, ExplainerMixin
from ..transform import SAX
from ..utils.validation import check_array, check_option

Importance = namedtuple("Importance", ["mean", "std", "full"])


def plot_importances(importances, ax=None, labels=None):
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


try:
    from matplotlib.cm import ScalarMappable, get_cmap
    from matplotlib.pylab import subplots

    from ..utils.plot import MidpointNormalize, plot_frequency_domain, plot_time_domain
except ModuleNotFoundError as e:
    from ..utils import DependencyMissing

    matplotlib_missing = DependencyMissing(e, package="matplotlib")
    get_cmap = matplotlib_missing
    ScalarMappable = matplotlib_missing
    subplots = matplotlib_missing
    MidpointNormalize = matplotlib_missing
    plot_frequency_domain = matplotlib_missing
    plot_time_domain = matplotlib_missing


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


class Domain(metaclass=ABCMeta):
    def intervals(self, n, n_intervals):
        return _intervals(n, n_intervals)

    @abstractmethod
    def plot_samples(self, x, y=None, ax=None, n_samples=None, **kwargs):
        pass

    @abstractmethod
    def transform(self, x):
        pass

    @abstractmethod
    def inverse_transform(self, x):
        pass

    @abstractmethod
    def randomize(self, x, start, end, random_state=None):
        pass


class TimeDomain(Domain):
    def plot_samples(self, x, y=None, ax=None, n_samples=None, **kwargs):
        plot_time_domain(x, y=y, ax=ax, n_samples=n_samples)

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x

    def randomize(self, x, start, end, random_state=None):
        if random_state is None:
            random_state = np.random
        random_state.shuffle(x[:, start:end])


class FrequencyDomain(Domain):
    def plot_samples(self, x, y=None, ax=None, n_samples=None, **kwargs):
        jitter = kwargs.pop("jitter", False)
        sample_spacing = kwargs.pop("sample_spacing", 1)
        frequency = kwargs.pop("show_frequency", False)
        plot_frequency_domain(
            x,
            y=y,
            ax=ax,
            n_samples=n_samples,
            jitter=jitter,
            sample_spacing=sample_spacing,
            frequency=frequency,
        )

    def intervals(self, n, n_intervals):
        for start, end in super().intervals(int(n // 2), n_intervals):
            yield start + 1, end + 1

    def transform(self, x):
        return np.fft.fft(x, axis=1)

    def inverse_transform(self, x):
        return np.fft.ifft(x, axis=1).real

    def randomize(self, x, start, end, random_state=None):
        if random_state is None:
            random_state = np.random
        random_state.shuffle(x[:, start:end])
        random_state.shuffle(x[:, end - 1 : start - 1 : -1])


_PERMUTATION_DOMAIN = {
    "time": TimeDomain,
    "frequency": FrequencyDomain,
}


class IntervalImportance(ExplainerMixin, BaseEstimator):
    """Compute a model agnostic importance score for non-overlapping intervals in
    the time or frequency domain by permuting the intervals among samples.

    Attributes
    ----------

    importances_ : dict or Importance
        The importance scores for each interval. If dict, one value per scoring
        function.

    """

    def __init__(
        self,
        *,
        scoring=None,
        n_repeat=5,
        n_intervals="sqrt",
        window=None,
        domain="time",
        verbose=False,
        random_state=None,
    ):
        """
        Parameters
        ----------
        scoring : str, list, dict or callable, optional
            The scoring function. By default the estimators score function is used.

        n_repeat : int, optional
            The number of repeated permutations

        n_intervals : str, optional
            The number of intervals.

            - if "sqrt", the number of intervals is the square root of n_timestep.

            - if "log", the number of intervals is the log2 of n_timestep.

            - if int, exact number of intervals.

        window : int, optional
            The window size. If specicied, n_intervals is ignored and the number of
            intervals is computed such that each interval is (at least) of size window.

        domain : {"time", "frequency"}, optional
            Compute the importance in the time or frequency domain.

        verbose : bool, optional
            Show extra progress information.

        random_state : int or RandomState
            - If `int`, `random_state` is the seed used by the random number generator
            - If `RandomState` instance, `random_state` is the random number generator
            - If `None`, the random number generator is the `RandomState` instance used
              by `np.random`.
        """
        self.scoring = scoring
        self.n_repeat = n_repeat
        self.n_intervals = n_intervals
        self.window = window
        self.domain = domain
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, estimator, x, y, sample_weight=None):
        estimator = self._validate_estimator(estimator)
        x, y = self._validate_data(x, y, reset=False, allow_3d=False)
        random_state = check_random_state(self.random_state)

        if self.window is not None:
            n_intervals = x.shape[-1] // check_scalar(
                self.window,
                "window",
                numbers.Integral,
                min_val=1,
                max_val=self.n_timesteps_in_,
            )
        elif self.n_intervals == "sqrt":
            n_intervals = math.ceil(math.sqrt(x.shape[-1]))
        elif self.n_intervals == "log":
            n_intervals = math.ceil(math.log2(x.shape[-1]))
        elif isinstance(self.n_intervals, numbers.Integral):
            n_intervals = self.n_intervals
        elif isinstance(self.n_intervals, numbers.Real):
            check_scalar(
                self.n_intervals,
                "n_intervals",
                numbers.Real,
                min_val=0.0,
                max_val=1.0,
                include_boundaries="right",
            )
            n_intervals = math.ceil(x.shape[-1] * self.n_intervals)
        else:
            raise ValueError(
                "n_intervals should either be 'sqrt', 'log', float or int, got %r"
                % self.n_intervals
            )

        if callable(self.scoring):
            scoring = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scoring = check_scoring(estimator, self.scoring)
        else:
            scoring_dict = _check_multimetric_scoring(estimator, self.scoring)
            scoring = _MultimetricScorer(**scoring_dict)

        if isinstance(self.domain, str):
            self.domain_ = check_option(_PERMUTATION_DOMAIN, self.domain, "domain")()
        else:
            self.domain_ = self.domain

        x_transform = self.domain_.transform(x=x)
        self.intervals_ = list(
            self.domain_.intervals(x_transform.shape[-1], n_intervals)
        )
        scores = []
        for iter, (start, end) in enumerate(self.intervals_):
            if self.verbose:
                print(f"Running iteration {iter + 1} of {len(self.intervals_)}.")
            x_perm_transform = x_transform.copy()
            rep_scores = []
            for rep in range(self.n_repeat):
                self.domain_.randomize(
                    x_perm_transform, start, end, random_state=random_state
                )
                x_perm_inverse = self.domain_.inverse_transform(x_perm_transform)
                if sample_weight is not None:
                    score = scoring(
                        estimator, x_perm_inverse, y, sample_weight=sample_weight
                    )
                else:
                    score = scoring(estimator, x_perm_inverse, y)
                rep_scores.append(score)

            if isinstance(rep_scores[0], dict):
                scores.append(_aggregate_score_dicts(rep_scores))
            else:
                scores.append(rep_scores)

        if sample_weight is not None:
            self.baseline_score_ = scoring(estimator, x, y, sample_weight=sample_weight)
        else:
            self.baseline_score_ = scoring(estimator, x, y)

        if self.verbose:
            print(f"Baseline score is: {self.baseline_score_}")

        if isinstance(self.baseline_score_, dict):
            self.importances_ = {
                name: _unpack_scores(
                    self.baseline_score_[name],
                    np.array([scores[i][name] for i in range(n_intervals)]),
                )
                for name in self.baseline_score_
            }
        else:
            self.importances_ = _unpack_scores(self.baseline_score_, np.array(scores))
        return self

    def explain(self, x, y=None):
        x = self._validate_data(x, reset=False)
        importances = np.empty(self.n_timesteps_in_, dtype=float)
        for i, (start, end) in enumerate(self.intervals_):
            importances[start:end] = self.importances_.mean[i]
        return np.broadcast_to(importances, (x.shape[0], self.n_timesteps_in_))

    def plot(
        self,
        x=None,
        y=None,
        *,
        ax=None,
        scoring=None,
        k=None,
        n_samples=100,
        show_grid=True,
        domain_args=None,
    ):
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
                labels=["(%d, %d)" % (start, end) for start, end in self.intervals_],
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
            vmin=importances[order[-1]],
            vmax=importances[order[0]],
            midpoint=0.0,
        )
        cmap = get_cmap("coolwarm")

        if show_grid:
            for start, _ in self.intervals_:
                ax.axvline(
                    start,
                    0,
                    1,
                    color="gray",
                    linestyle="dashed",
                    linewidth=0.5,
                    dashes=(5, 5),
                    zorder=-100,
                )

        for o in order:
            start, end = self.intervals_[o]
            ax.axvspan(
                start,
                end,
                0,
                1,
                alpha=0.2,
                color=cmap(norm(importances[o])),
                zorder=100,
            )

        xticks = [start for start, _ in self.intervals_]
        xticks.append(self.intervals_[-1][1])
        ax.set_xticks(xticks)
        ax.tick_params(axis="x", labelrotation=-70)

        self.domain_.plot_samples(
            x,
            y=y,
            ax=ax,
            n_samples=n_samples,
            **(domain_args if domain_args is not None else {}),
        )

        mappable = ScalarMappable(cmap=cmap, norm=norm)
        if fig is not None:
            fig.colorbar(mappable)
            return ax
        else:
            return ax, mappable


class AmplitudeImportance(ExplainerMixin, BaseEstimator):
    """Compute the importance of equi-probable horizontal time series intervals by
    permuting the values between each horizontal interval. The implementation uses
    :class:`transform.SAX` to discretize the time series and then for each bin permute
    the samples along that bin.

    Paramters
    ---------

    sax_ : SAX
        The SAX transformation.

    baseline_score_ : float or dict
        The baseline score.

    importances_ : float or dict
        The importances of each vertical bin.

    """

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
        self.scoring = scoring
        self.n_intervals = n_intervals
        self.window = window
        self.binning = binning
        self.n_bins = n_bins
        self.n_repeat = n_repeat
        self.random_state = random_state

    def fit(self, estimator, x, y, sample_weight=None):
        estimator = self._validate_estimator(estimator)
        x, y = self._validate_data(x, y, reset=False, allow_3d=False)

        self.sax_ = SAX(
            n_intervals=self.n_intervals,
            window=self.window,
            binning=self.binning,
            n_bins=self.n_bins,
            estimate=True,
        ).fit(x)

        x_sax = self.sax_.transform(x)

        if callable(self.scoring):
            scoring = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scoring = check_scoring(estimator, self.scoring)
        else:
            scoring_dict = _check_multimetric_scoring(estimator, self.scoring)
            scoring = _MultimetricScorer(**scoring_dict)

        random_state = check_random_state(self.random_state)
        scores = []
        for bin in range(self.n_bins):
            rep_scores = []
            for rep in range(self.n_repeat):
                x_perm = x.copy()
                for i in range(x_sax.shape[0] - 1, 0, -1):
                    j = random_state.randint(0, i + 1)
                    xi = x_sax[i, :]
                    xj = x_sax[j, :]

                    i_indicies = np.where(xi == bin)[0]
                    j_indicies = np.where(xj == bin)[0]

                    if i_indicies.size < j_indicies.size:
                        j_indicies = random_state.choice(
                            j_indicies, i_indicies.size, replace=False
                        )
                    elif i_indicies.size > j_indicies.size:
                        i_indicies = random_state.choice(
                            i_indicies, j_indicies.size, replace=False
                        )

                    for i_idx, j_idx in zip(i_indicies, j_indicies):
                        i_start, i_end = self.sax_.intervals[i_idx]
                        j_start, j_end = self.sax_.intervals[j_idx]

                        if i_end - i_start < j_end - j_start:
                            j_end -= 1
                        elif j_end - j_start < i_end - i_start:
                            i_end -= 1

                        x_perm[i, i_start:i_end], x_perm[j, j_start:j_end] = (
                            x_perm[j, j_start:j_end],
                            x_perm[i, i_start:i_end],
                        )

                if sample_weight is not None:
                    score = scoring(estimator, x_perm, y, sample_weight=sample_weight)
                else:
                    score = scoring(estimator, x_perm, y)

                rep_scores.append(score)

            if isinstance(rep_scores[0], dict):
                scores.append(_aggregate_score_dicts(rep_scores))
            else:
                scores.append(rep_scores)

        if sample_weight is not None:
            self.baseline_score_ = scoring(estimator, x, y, sample_weight=sample_weight)
        else:
            self.baseline_score_ = scoring(estimator, x, y)

        if isinstance(self.baseline_score_, dict):
            self.importances_ = {
                name: _unpack_scores(
                    self.baseline_score_[name],
                    np.array([scores[i][name] for i in range(self.n_bins)]),
                )
                for name in self.baseline_score_
            }
        else:
            self.importances_ = _unpack_scores(self.baseline_score_, np.array(scores))

        return self

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

    def plot(
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
        """Plot the importances. If x is given, the importances are plotted over the
        samples optionally labeling each sample using the supplied labels. If x is
        not give, the importances are plotted as one or more boxplots.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_timesteps), optional
            The samples

        y : array-like of shape (n_samples, ), optional
            The labels

        ax : Axes, optional
            Axes to plot. If ax is set, x is None and scoring is None, the number of
            axes must be the same as the number of scorers.

        scoring : str, optional
            The scoring to plot if multiple scorers were used when fitting.

        preprocess : bool, optional
            Preprocess the time series to align with the bins, ignored if x is not None.

        k : int or float, optional
            The number of top bins to plot, ignored if x is not None.

            - if int, the specified number of bins are shown
            - if float, a fraction of the number of bins are shown

        show_bins : bool, optional
            Annotate the plot with the index of the bin, ignored if x is not None.

        show_grid : bool, optional
            Annotate the plot with the bin thresholds, ignored if x is not None.

        Returns
        -------
        ax : Axis
            The axis

        mappable : ScalarMappable, optional
            Return the mappable used to plot the colorbar.
            Only returned if ax is not None and x is not None.
        """
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
            vmin=importances[order[-1]], vmax=importances[order[0]], midpoint=0.0
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
                    zorder=-100,
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
                zorder=100,
                color=cmap(norm(importances[i])),
                label="%i" % i,
            )

            if show_bins:
                ax.annotate("%d" % i, xy=(0, 0.5), va="center", xycoords=span)

        plot_time_domain(x, y, ax=ax, n_samples=n_samples)
        ax.set_ylim([np.min(x) - np.std(x), np.max(x) + np.std(x)])
        mappable = ScalarMappable(cmap=cmap, norm=norm)
        if fig is not None:
            fig.colorbar(mappable)
            return ax
        else:
            return ax, mappable
