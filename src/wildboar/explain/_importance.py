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
from sklearn.utils.validation import check_is_fitted, check_random_state, check_scalar

from ..base import BaseEstimator, ExplainerMixin
from ..distance import pairwise_subsequence_distance
from ..transform import SAX, RandomShapeletTransform
from ..utils.validation import check_array, check_option

try:
    from matplotlib.cm import ScalarMappable, get_cmap
    from matplotlib.pylab import subplots

    from ..utils.plot import MidpointNormalize, plot_time_domain
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
    """Plot the importances as a boxplot.

    Paramters
    ---------
    importances : Importance or dict
        The importances

    ax : Axes, optional
        The axes to plot. If importances is dict, ax must contain at least
        len(importances) Axes objects.

    labels : array-like, optional
        The labels for the importances.

    Returns
    -------
    ax : Axes
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
    def __init__(self, *, scoring=None, n_repeat=1, verbose=0, random_state=None):
        self.scoring = scoring
        self.n_repeat = n_repeat
        self.random_state = random_state
        self.verbose = verbose

    @abstractmethod
    def _yield_components(self):
        pass

    def _reset_component(self, X, component):
        pass

    @abstractmethod
    def _permute_component(self, X, component, random_state):
        pass

    def _fit(self, estimator, X, y, random_state, sample_weight=None):
        if callable(self.scoring):
            scoring = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scoring = check_scoring(estimator, self.scoring)
        else:
            scoring_dict = _check_multimetric_scoring(estimator, self.scoring)
            scoring = _MultimetricScorer(**scoring_dict)

        self.components_ = np.array(list(self._yield_components()), dtype=object)
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
    """Compute a model agnostic importance score for non-overlapping intervals

    Attributes
    ----------

    importances_ : dict or Importance
        The importance scores for each interval. If dict, one value per scoring
        function.

    components_ : ndarray of shape (n_intervals, 2)
        The interval start and end positions.

    """

    def __init__(
        self,
        *,
        scoring=None,
        n_repeat=5,
        n_intervals="sqrt",
        window=None,
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
        super().__init__(scoring=scoring, n_repeat=n_repeat, random_state=random_state)
        self.n_intervals = n_intervals
        self.window = window
        self.verbose = verbose

    def _yield_components(self):
        if self.window is not None:
            n_intervals = self.n_timesteps_in_ // check_scalar(
                self.window,
                "window",
                numbers.Integral,
                min_val=1,
                max_val=self.n_timesteps_in_,
            )
        elif self.n_intervals == "sqrt":
            n_intervals = math.ceil(math.sqrt(self.n_timesteps_in_))
        elif self.n_intervals == "log":
            n_intervals = math.ceil(math.log2(self.n_timesteps_in_))
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
            n_intervals = math.ceil(self.n_timesteps_in_ * self.n_intervals)
        else:
            raise ValueError(
                "n_intervals should either be 'sqrt', 'log', float or int, got %r"
                % self.n_intervals
            )
        return _intervals(self.n_timesteps_in_, n_intervals)

    def _permute_component(self, X, component, random_state):
        X_perm = X.copy()
        start, end = component
        random_state.shuffle(X_perm[:, start:end])
        return X_perm

    def fit(self, estimator, x, y, sample_weight=None):
        estimator = self._validate_estimator(estimator)
        x, y = self._validate_data(x, y, reset=False, allow_3d=False)
        random_state = check_random_state(self.random_state)
        self._fit(estimator, x, y, random_state, sample_weight=sample_weight)
        return self

    def explain(self, x, y=None):
        check_is_fitted(self)
        x = self._validate_data(x, reset=False)
        importances = np.empty(self.n_timesteps_in_, dtype=float)
        for i, (start, end) in enumerate(self.components_):
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
            vmin=importances[order[-1]],
            vmax=importances[order[0]],
            midpoint=0.0,
        )
        cmap = get_cmap("coolwarm")

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
            fig.colorbar(mappable)
            return ax
        else:
            return ax, mappable


class AmplitudeImportance(ExplainerMixin, PermuteImportance):
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

    components_ : list
        List of binning identifiers.

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
        super().__init__(scoring=scoring, n_repeat=n_repeat, random_state=random_state)
        self.n_intervals = n_intervals
        self.window = window
        self.binning = binning
        self.n_bins = n_bins

    def _yield_components(self):
        return (b for b in range(self.n_bins))

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
        estimator = self._validate_estimator(estimator)
        X, y = self._validate_data(X, y, reset=False, allow_3d=False)
        self.sax_ = SAX(
            n_intervals=self.n_intervals,
            window=self.window,
            binning=self.binning,
            n_bins=self.n_bins,
            estimate=True,
        ).fit(X)
        self.x_sax_ = self.sax_.transform(X)
        random_state = check_random_state(self.random_state)
        self._fit(estimator, X, y, random_state, sample_weight=sample_weight)
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
            fig.colorbar(mappable)
            return ax
        else:
            return ax, mappable


class ShapeletImportance(ExplainerMixin, PermuteImportance):
    """Compute the importance of shapelets by permuting instances with lowest distance.

    Attributes
    ----------
    components : ndarray
        The shapelets
    """

    def __init__(
        self,
        scoring=None,
        n_repeat=1,
        n_shapelets=10,
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        metric="euclidean",
        metric_params=None,
        random_state=None,
    ):
        super().__init__(scoring=scoring, n_repeat=n_repeat, random_state=random_state)
        self.n_shapelets = n_shapelets
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.metric = metric
        self.metric_params = metric_params

    def _reset_component(self, X, component):
        _, self._shapelet_idx = pairwise_subsequence_distance(
            component,
            X,
            metric=self.metric,
            metric_params=self.metric_params,
            return_index=True,
        )

    def _yield_components(self):
        for _dim, (_, shapelet) in self.shapelet_transform_.embedding_.features:
            yield shapelet

    def _permute_component(self, X, component, random_state):
        X_perm = np.empty_like(X)
        idx = np.arange(X.shape[0])
        random_state.shuffle(idx)
        for i in range(X.shape[0]):
            j = idx[i]

            i_shapelet_idx = self._shapelet_idx[i]
            j_shapelet_idx = self._shapelet_idx[j]
            X_perm[j] = X[j]
            if i != j:
                X_perm[j, j_shapelet_idx : (j_shapelet_idx + component.size)] = X[
                    i, i_shapelet_idx : (i_shapelet_idx + component.size)
                ]

        return X_perm

    def fit(self, estimator, X, y, sample_weight=None):
        self._validate_estimator(estimator)
        X, y = self._validate_data(X, y, reset=False)

        random_state = check_random_state(self.random_state)
        self.shapelet_transform_ = RandomShapeletTransform(
            n_shapelets=self.n_shapelets,
            metric=self.metric,
            min_shapelet_size=self.min_shapelet_size,
            max_shapelet_size=self.max_shapelet_size,
            random_state=random_state.randint(np.iinfo(np.int32).max),
        ).fit(X)

        self._fit(estimator, X, y, random_state, sample_weight=sample_weight)
        return self

    def explain(self, X, y=None, kernel_scale=0.25):
        distances, indices = pairwise_subsequence_distance(
            self.components_,
            X,
            metric=self.metric,
            metric_params=self.metric_params,
            return_index=True,
        )

        weights = self._distance_weight(distances, kernel_scale)
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

    def plot(self, X=None, y=None, k=None, scoring=None, kernel_scale=0.25, ax=None):
        if X is None:
            plot_importances(
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
            explanation = self.explain(X, kernel_scale=kernel_scale)
            if ax is None:
                fig, ax = subplots()
            else:
                fig = None

            mappable = ax.pcolormesh(explanation[order[:k]], cmap="coolwarm")
            ax.set_yticks(np.arange(k) + 0.5, order[:k])
            if fig is not None:
                fig.colorbar(mappable)
                return ax
            else:
                return ax, mappable
        else:
            explanation = self.explain(X, y, kernel_scale=kernel_scale)
            distances, index = pairwise_subsequence_distance(
                self.components_,
                X,
                metric=self.metric,
                metric_params=self.metric_params,
                return_index=True,
            )
            cmap = get_cmap("coolwarm")
            norm = MidpointNormalize(
                vmin=importances[:k].min(), vmax=importances[:k].max(), midpoint=0
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

            weights = self._distance_weight(distances, 0.25)
            for i in range(len(labels)):
                for j in range(k):
                    plot_time_domain(
                        X[y == labels[i]], n_samples=lbl_count[i], ax=ax[j, i]
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

            mappable = ScalarMappable(norm=norm, cmap=cmap)
            if ax is not None:
                fig.colorbar(mappable, orientation="horizontal", ax=ax[-1, :])
                return ax
            else:
                return ax, mappable

    def _distance_weight(self, distances, kernel_scale=0.25):
        kernel_width = [np.sqrt(s.size) * kernel_scale for s in self.components_]
        return np.sqrt(np.exp(-(distances**2) / np.array(kernel_width) ** 2))
