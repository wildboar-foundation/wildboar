# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np
from sklearn.tree._reingold_tilford import Tree, buchheim
from sklearn.utils import Bunch, column_or_1d, resample

from ..transform._interval import IntervalMixin
from ..transform._shapelet import ShapeletMixin
from ..utils.validation import check_array

try:
    import matplotlib.pylab as plt
    from matplotlib.cm import get_cmap
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
except ModuleNotFoundError as e:
    from ..utils import DependencyMissing

    matplotlib_missing = DependencyMissing(e, package="matplotlib")
    plt = matplotlib_missing
    get_cmap = matplotlib_missing
    Normalize = matplotlib_missing
    LineCollection = matplotlib_missing
    Line2D = matplotlib_missing

__all__ = ["plot_tree", "plot_time_domain", "plot_frequency_domain"]


def plot_tree(
    clf,
    ax,
    bbox_args=dict(),
    arrow_args=dict(arrowstyle="<-"),
    max_depth=None,
    show_classes=True,
    node_labeler=None,
):
    def _add_attributes(tree, attributes):
        setattr(tree, "_attr", attributes)

        return tree

    def node_attributes(tree, node_id, proportion=True, precision=2, class_names=None):
        attribute = tree.attribute[node_id]
        if attribute is None:
            dim = None
            attribute = None
        else:
            dim, attribute = attribute

        return Bunch(
            node_id=node_id,
            value=tree.value[node_id],
            threshold=tree.threshold[node_id],
            dim=dim,
            attribute=attribute,
            n_node_samples=tree.n_node_samples[node_id],
        )

    def _make_tree(node_id, et, depth=0):
        if et.left[node_id] != -1 and (max_depth is None or depth <= max_depth):
            children = [
                _make_tree(et.left[node_id], et, depth=depth + 1),
                _make_tree(et.right[node_id], et, depth=depth + 1),
            ]
            tree = Tree(str(node_id), node_id, *children)
        else:
            tree = Tree(str(node_id), node_id)

        setattr(tree, "_attr", node_attributes(et, node_id))
        return tree

    if node_labeler is None:

        def _prediction(value):
            if show_classes and hasattr(clf, "classes_"):
                return clf.classes_[np.argmax(value)]
            elif value.shape[0] == 1:
                return np.round(value[0], 2)
            else:
                return np.round(value, 2)

        if isinstance(clf, ShapeletMixin):

            def node_labeler(tree):
                if tree.children:
                    x_label = "X"
                    if clf.n_dims_in_ > 1:
                        x_label = "X[%d]" % tree._attr.dim
                    return "D(%s, S[%i]) <= %.2f\n%d samples" % (
                        x_label,
                        tree._attr.node_id,
                        tree._attr.threshold,
                        tree._attr.n_node_samples,
                    )
                else:
                    return "Pred: %s\n%d samples" % (
                        _prediction(tree._attr.value),
                        tree._attr.n_node_samples,
                    )
        elif isinstance(clf, IntervalMixin):

            def node_labeler(tree):
                if tree.children:
                    (start, length, output) = tree._attr.attribute
                    x_label = "X[%d:%d]" % (start, start + length)
                    if clf.n_dims_in_ > 1:
                        x_label = "X[%d, %d:%d]" % (
                            tree._attr.dim,
                            start,
                            start + length,
                        )
                    return "%s(%s) <= %.2f\n%d samples" % (
                        clf.tree_.generator.summarizer.value_name(output),
                        x_label,
                        tree._attr.threshold,
                        tree._attr.n_node_samples,
                    )
                else:
                    return "Pred: %s\n%d samples" % (
                        _prediction(tree._attr.value),
                        tree._attr.n_node_samples,
                    )
        else:

            def node_labeler(tree):
                return str(tree._attr["node_id"])

    draw_tree = buchheim(_make_tree(0, clf.tree_))
    max_x, max_y = draw_tree.max_extents() + 1

    def recurse(node, depth):
        kwargs = dict(
            bbox=bbox_args.copy(),
            ha="center",
            va="center",
            zorder=100 - 10 * depth,
            xycoords="axes fraction",
            arrowprops=arrow_args.copy(),
        )
        kwargs["arrowprops"]["edgecolor"] = plt.rcParams["text.color"]

        # offset things by .5 to center them in plot
        xy = ((node.x + 0.5) / max_x, (max_y - node.y - 0.5) / max_y)
        x_loc, y_loc = xy
        if max_depth is None or depth <= max_depth:
            kwargs["bbox"]["fc"] = ax.get_facecolor()

            label = node_labeler(node.tree)
            if node.parent is None:
                ax.annotate(label, xy, **kwargs)
            else:
                xy_parent = (
                    (node.parent.x + 0.5) / max_x,
                    (max_y - node.parent.y - 0.5) / max_y,
                )
                ax.annotate(label, xy_parent, xy, **kwargs)

            for child in node.children:
                recurse(child, depth + 1)

        else:
            xy_parent = (
                (node.parent.x + 0.5) / max_x,
                (max_y - node.parent.y - 0.5) / max_y,
            )
            kwargs["bbox"]["fc"] = "grey"
            ax.annotate("\n  (...)  \n", xy_parent, xy, **kwargs)

    recurse(draw_tree, 0)
    return ax


# noqa: H0002
class MidpointNormalize(Normalize):
    """
    Normalise values with a specified midpoint.

    Parameters
    ----------
    vmin : float, optional
        The minumum value.
    vmax : float, optional
        The maximum value.
    midpoint : float, optional
        The midpoint of the values.
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None):
        super().__init__(vmin, vmax, clip=False)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def plot_time_domain(
    x,
    y=None,
    *,
    n_samples=100,
    ax=None,
    alpha=0.5,
    linewidth=0.5,
    zorder=-1,
    cmap="Dark2",
    show_legend=8,
):
    """
    Plot the samples in the time domain.

    Parameters
    ----------
    x : array-like of shape (n_sample, n_timestep)
        The samples.
    y : array-like of shape (n_samples, ), optional
        The labels, assumed to be discrete labels.
    n_samples : int, optional
        The maximum number of samples to plot. If n_samples is larger than the number
        of samples in x or None, all samples are plotted.
    ax : Axes, optional
        The matplotlib Axes-object.
    alpha : float, optional
        The opacity of the samples.
    linewidth : float, optional
        The width of the sample lines.
    zorder : int, optional
        The order where the samples are plotted. By default we plot the samples
        at -1.
    cmap : str, optional
        The colormap used to colorize samples according to its label.
    show_legend : bool or int, optional
        Whether the legend of labels are show.

        - if bool, show the legend if y is not None
        - if int, show the legend if the number of labels are lower than the
          show_legend parameter value

    Returns
    -------
    Axes
        The axes object that has been plotted.

    Examples
    --------
    >>> from wildboar.utils.plot import plot_time_domain
    >>> from wildboar.datasets import load_gun_point
    >>> X, y = load_gun_point(X, y)
    >>> plot_time_domain(X, y, n_sample=10)
    """
    if ax is None:
        fig, ax = plt.subplots()

    x = check_array(np.atleast_2d(x), input_name="x", allow_3d=False, order=None)
    if y is not None:
        y = column_or_1d(y, warn=True)
        if y.shape[0] != x.shape[0]:
            raise ValueError("The number of labels and samples are not the same.")

        labels, idx, inv = np.unique(y, return_inverse=True, return_index=True)
        cmap = get_cmap(cmap, len(labels))
    else:
        cmap = get_cmap(cmap, 1)
        inv = np.zeros(x.shape[0])

    if n_samples is not None:
        x, inv = resample(
            x,
            inv,
            replace=False,
            n_samples=n_samples if n_samples < x.shape[0] else x.shape[0],
            stratify=y,
            random_state=0,
        )

    x_axis = np.arange(x.shape[-1] + 1)
    collection = LineCollection(
        [list(zip(x_axis, x[i])) for i in range(x.shape[0])],
        colors=[cmap(inv[i]) for i in range(x.shape[0])],
        zorder=zorder,
        linewidth=linewidth,
        alpha=alpha,
    )
    ax.add_collection(collection)

    if y is not None and (show_legend is True or len(labels) <= show_legend):
        legend = ax.legend(
            [Line2D([0], [0], color=cmap(i)) for i in idx],
            labels,
            loc="best",
            ncol=(len(labels) // 3) + 1,
        )
        legend.set_zorder(100)
        ax.add_artist(legend)

    ax.set_xlim([1, x.shape[-1] - 1])
    ax.set_ylim([np.min(x) - np.std(x), np.max(x) + np.std(x)])

    return ax


def plot_frequency_domain(
    x,
    y=None,
    *,
    ax=None,
    n_samples=100,
    jitter=False,
    sample_spacing=1,
    frequency=False,
    cmap="Dark2",
):
    """
    Plot the samples in the freqency domain.

    Parameters
    ----------
    x : array-like of shape (n_sample, n_timestep)
        The samples.
    y : array-like of shape (n_samples, ), optional
        The labels.
    ax : Axes, optional
        The matplotlib Axes-object.
    n_samples : int, optional
        The maximum number of samples to plot. If n_samples is larger than the number
        of samples in x or None, all samples are plotted.
    jitter : bool, optional
        Add jitter to the amplitude lines.
    sample_spacing : int, optional
        The frequency domain sample spacing.
    frequency : bool, optional
        Show the frequency bins.
    cmap : str, optional
        The colormap.

    Returns
    -------
    Axes
        The axes object that has been plotted.

    Examples
    --------
    >>> from wildboar.utils.plot import plot_frequency_domain
    >>> from wildboar.datasets import load_gun_point
    >>> X, y = load_gun_point(X, y)
    >>> plot_frequency_domain(X, y, n_sample=10)
    """
    if ax is None:
        fig, ax = plt.subplots()

    x = check_array(x, allow_3d=False, order=None)
    if y is not None:
        y = check_array(y, ensure_2d=False, allow_nd=False, order=None)
        if y.shape[0] != x.shape[0]:
            raise ValueError("The number of samples and lables are not the same.")

        labels, idx, inv = np.unique(y, return_inverse=True, return_index=True)
        cmap = get_cmap(cmap, len(labels))
    else:
        cmap = get_cmap(cmap, 1)
        inv = np.zeros(x.shape[0])

    x, inv = resample(
        x,
        inv,
        replace=False,
        n_samples=n_samples if n_samples < x.shape[0] else x.shape[0],
        stratify=y,
        random_state=0,
    )

    n_freqs = int(x.shape[-1] // 2)
    x_freq = np.abs(np.fft.fft(x, axis=1)[:, 1 : n_freqs + 1])  # / n_freqs
    x_axis = np.arange(1, n_freqs + 1)
    max_freq = np.max(x_freq)
    if frequency:
        for i in x_axis:
            ax.axvspan(
                i - 0.5,
                i + 0.5,
                0,
                1,
                facecolor=None,
                edgecolor="gray",
                fill=False,
                alpha=0.05,
                zorder=-100,
            )
    ax.set_ylabel("Amplitude")
    for i in range(x.shape[0]):
        if jitter:
            x_axis_tmp = x_axis + np.random.normal(scale=0.5, size=n_freqs)
        else:
            x_axis_tmp = x_axis

        ax.vlines(
            x_axis_tmp,
            0,
            x_freq[i],
            color=cmap(inv[i]),
            alpha=0.3,
            linewidth=1,
            zorder=-1,
        )

    if y is not None:
        legend = ax.legend(
            [Line2D([0], [0], color=cmap(i)) for i in idx],
            labels,
            loc="best",
            ncol=(len(labels) // 3) + 1,
        )

        ax.add_artist(legend)

    # ticks = ax.get_xticks().astype(int)[: len(x_axis)]
    # ticks[0] = 1
    # ax.set_xticks(ticks)
    # x_label = np.fft.fftfreq(x.shape[-1], d=sample_spacing)[ticks]
    # ax.set_xticklabels("%.2f" % lab for lab in x_label)
    ax.set_xlim([0.5, n_freqs + 0.5])
    ax.set_ylim(0, max_freq + np.std(x_freq))

    return ax
