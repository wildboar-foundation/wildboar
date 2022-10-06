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

from wildboar.utils.validation import check_option

from ..base import BaseEstimator, ExplainerMixin

Importance = namedtuple("Importance", ["mean", "std", "full"])


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
    def plot_samples(self, x, y=None, ax=None, **kwargs):
        plot_time_domain(x, y=y, ax=ax)

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x

    def randomize(self, x, start, end, random_state=None):
        if random_state is None:
            random_state = np.random
        random_state.shuffle(x[:, start:end])


class FrequencyDomain(Domain):
    def plot_samples(self, x, y=None, ax=None, **kwargs):
        jitter = kwargs.pop("jitter", False)
        sample_spacing = kwargs.pop("sample_spacing", 1)
        frequency = kwargs.pop("frequency", False)
        plot_frequency_domain(
            x,
            y=y,
            ax=ax,
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
        self.domain = domain
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, estimator, x, y, sample_weight=None):
        estimator = self._validate_estimator(estimator)
        x, y = self._validate_data(x, y, reset=False, allow_3d=False)
        random_state = check_random_state(self.random_state)

        if self.n_intervals == "sqrt":
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
        top_k=None,
        n_samples=None,
        **kwargs,
    ):
        if isinstance(self.importances_, dict):
            if scoring is None and scoring not in self.importances_:
                raise ValueError(
                    "scoring must be %s, got %r"
                    % (set(self.importances_.keys()), scoring)
                )
            importances = self.importances_[scoring].mean
        else:
            if scoring is not None and scoring != self.scoring:
                raise ValueError("scoring must be %s, got %r" % (self.scoring, scoring))
            importances = self.importances_.mean

        if top_k is None or top_k > importances.shape[0]:
            top_k = importances.shape[0]
        elif isinstance(top_k, numbers.Real):
            top_k = check_scalar(
                top_k,
                "top_k",
                numbers.Real,
                min_val=0,
                max_val=1,
                include_boundaries="right",
            )
            top_k = math.ceil(top_k * importances.shape[0])

        if ax is None:
            fig, ax = subplots()
        else:
            fig = None

        order = np.argsort(importances)[: -(top_k + 1) : -1]
        norm = MidpointNormalize(
            vmin=max(importances[order[-1]] - 0.1, 0),
            vmax=min(importances[order[0]] + 0.1, 1),
            midpoint=0.0,
        )
        cmap = get_cmap("coolwarm")
        if x is not None:
            if n_samples is None:
                x = x
            else:
                if isinstance(n_samples, numbers.Real):
                    n_samples = math.ceil(
                        x.shape[0]
                        * check_scalar(
                            n_samples,
                            "n_sample",
                            numbers.Real,
                            min_val=0,
                            max_val=1,
                            include_boundaries="right",
                        )
                    )
                else:
                    n_samples = n_samples

                x = x[:n_samples, :]
                if y is not None:
                    y = y[:n_samples]

            self.domain_.plot_samples(x, y=y, ax=ax, **kwargs)

        for o in order:
            start, end = self.intervals_[o]
            if end - start > 1:
                ax.axvspan(
                    start - 0.5,
                    end - 0.5,
                    0,
                    1,
                    alpha=0.3,
                    color=cmap(norm(importances[o])),
                    zorder=100,
                )
            else:
                ax.axvspan(
                    start - 0.5,
                    start + 0.5,
                    0,
                    1,
                    alpha=0.3,
                    color=cmap(norm(importances[o])),
                    zorder=100,
                )

        mappable = ScalarMappable(cmap=cmap, norm=norm)
        if fig is not None:
            fig.colorbar(mappable)
            return ax
        else:
            return ax, mappable
