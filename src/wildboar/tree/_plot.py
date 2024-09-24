# Authors: Isak Samsten
# License: BSD 3 clause
import numpy as np
from sklearn.tree._reingold_tilford import Tree, buchheim
from sklearn.utils import Bunch

try:
    import matplotlib.pylab as plt
    from matplotlib.text import Annotation
except ModuleNotFoundError as e:
    from ..utils import DependencyMissing

    matplotlib_missing = DependencyMissing(e, package="matplotlib")
    plt = matplotlib_missing
    Annotation = matplotlib_missing

__all__ = ["plot_tree"]


def plot_tree(
    clf,
    *,
    ax=None,
    bbox_args=dict(),
    arrow_args=dict(arrowstyle="<-"),
    max_depth=None,
    class_labels=True,
    fontsize=None,
    node_labeler=None,
):
    """
    Plot a tree

    Parameters
    ----------
    clf : tree-based estimator
        A decision tree.
    ax : axes, optional
        The axes to plot the tree to.
    bbox_args : dict, optional
        Arguments to the node box.
    arrow_args : dict, optional
        Arguments to the arrow.
    max_depth : int, optional
        Only show the branches until `max_depth`.
    class_labels : bool or array-like, optional
        Show the classes

        - if True, show classes from the `classes_` attribute of the decision
          tree.
        - if False, show leaf probabilities.
        - if array-like, show classes from the array.
    fontsize : int, optional
        The font size. If `None`, the font size is determined automatically.
    node_labeler : callable, optional
        A function returning the label for a node on the form `f(node) ->
        str)`.

        - If ``node.children is not None`` the node is a leaf.
        - ``node._attr`` contains information about the node:

          - ``n_node_samples``: the number of samples reaching the node
          - if leaf, ``value`` is an array with the fractions of labels
            reaching the leaf (in case of classification); or the mean among
            the samples reach the leaf (if regression). Determine if it is a
            classification or regression tree by inspecting the shape of the
            value array.
          - if branch, ``threshold`` contains the threshold used to split the
            node.
          - if branch, ``dim`` contains the dimension from which the attribute
            was extracted.
          - if branch, ``attribute`` contains the attribute used for computing
            the feature value. The attribute depends on the estimator.

    Returns
    -------
    axes
        The axes.

    Examples
    --------
    >>> from wildboar.datasets import load_two_lead_ecg
    >>> from wildboar.tree import ShapeletTreeClassifier, plot_tree
    >>> X, y = load_two_lead_ecg()
    >>> clf = ShapeletTreeClassifier(strategy="random").fit(X, y)
    >>> plot_tree(clf)
    <Axes: >

    """
    from ..transform._interval import IntervalMixin
    from ..transform._shapelet import ShapeletMixin

    def _add_attributes(tree, attributes):
        setattr(tree, "_attr", attributes)

        return tree

    def node_attributes(tree, node_id):
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
            if class_labels is not None and class_labels is not False:
                argmax = np.argmax(value)
                if class_labels is True and hasattr(clf, "classes_"):
                    return clf.classes_[argmax]
                else:
                    return class_labels[argmax]
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

    if ax is None:
        _, ax = plt.subplots()

    ax_width = ax.get_window_extent().width
    ax_height = ax.get_window_extent().height

    scale_x = ax_width / max_x
    scale_y = ax_height / max_y

    def _draw_tree_recursive(node, depth):
        kwargs = dict(
            bbox=bbox_args.copy(),
            ha="center",
            va="center",
            zorder=100 - 10 * depth,
            xycoords="axes fraction",
            arrowprops=arrow_args.copy(),
        )
        kwargs["arrowprops"]["edgecolor"] = plt.rcParams["text.color"]

        if fontsize is not None:
            kwargs["fontsize"] = fontsize

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
                _draw_tree_recursive(child, depth + 1)

        else:
            xy_parent = (
                (node.parent.x + 0.5) / max_x,
                (max_y - node.parent.y - 0.5) / max_y,
            )
            kwargs["bbox"]["fc"] = "grey"
            ax.annotate("\n  (...)  \n", xy_parent, xy, **kwargs)

    _draw_tree_recursive(draw_tree, 0)

    anns = [ann for ann in ax.get_children() if isinstance(ann, Annotation)]
    renderer = ax.figure.canvas.get_renderer()

    for ann in anns:
        ann.update_bbox_position_size(renderer)

    if fontsize is None:
        extents = [ann.get_bbox_patch().get_window_extent() for ann in anns]
        max_width = max([extent.width for extent in extents])
        max_height = max([extent.height for extent in extents])
        size = anns[0].get_fontsize() * min(scale_x / max_width, scale_y / max_height)
        for ann in anns:
            ann.set_fontsize(size)

    ax.set_axis_off()
    return ax
