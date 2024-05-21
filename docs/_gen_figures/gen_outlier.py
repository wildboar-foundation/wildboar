import matplotlib.pylab as plt
import numpy as np
from light_dark import if_not_exists, yield_and_save_plot
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from wildboar.datasets import load_dataset
from wildboar.datasets.outlier import (
    emmott_outliers,
    kmeans_outliers,
    majority_outliers,
    minority_outliers,
)
from wildboar.transform import IntervalTransform
from wildboar.utils import plot


def mkfigure():
    fig = plt.figure(figsize=(6, 3.72))
    gs = plt.GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    return fig, np.array([ax1, ax2, ax3])


def plot_inlier_outlier(
    ax, *, original_projection, outlier_projection, x_outlier, y_outlier
):
    label, value = np.unique(y_outlier, return_counts=True)
    ax[0].set_title("Projection")
    plot_projection(
        original_projection=original_projection,
        outlier_projection=outlier_projection,
        y_outlier=y_outlier,
        ax=ax[0],
    )

    ax[1].set_title("Distribution")

    ax[1].bar(label, value, color=plt.get_cmap(lut=2).colors)

    ax[2].set_title("Time series")
    plot.plot_time_domain(
        x_outlier, y=y_outlier, ax=ax[2], n_samples=10, cmap=plt.get_cmap(lut=2)
    )


def make_projection(*, x, x_outlier):
    pipe = make_pipeline(IntervalTransform(n_intervals=15), PCA(2, random_state=123))
    original_projection = pipe.fit_transform(x)
    outlier_projection = pipe.transform(x_outlier)
    return original_projection, outlier_projection


def plot_projection(*, original_projection, outlier_projection, y_outlier, ax):
    cmap = plt.get_cmap(lut=3)
    ax.scatter(
        original_projection[:, 0],
        original_projection[:, 1],
        color=cmap(1),
        alpha=0.2,
        marker="x",
    )
    ax.scatter(
        outlier_projection[y_outlier == 1, 0],
        outlier_projection[y_outlier == 1, 1],
        marker="x",
        color=cmap(2),
    )
    ax.scatter(
        outlier_projection[y_outlier == -1, 0],
        outlier_projection[y_outlier == -1, 1],
        marker="x",
        color=cmap(0),
    )


@if_not_exists(
    "guide",
    "unsupervised",
    "outlier",
    "minority.svg",
)
def gen_minority(path):
    x, y = load_dataset("CBF", progress=False)
    x_outlier, y_outlier = minority_outliers(x, y, n_outliers=0.05)
    original_projection, outlier_projection = make_projection(x=x, x_outlier=x_outlier)

    for fig, ax in yield_and_save_plot(path, figure=mkfigure):
        plot_inlier_outlier(
            ax,
            original_projection=original_projection,
            outlier_projection=outlier_projection,
            x_outlier=x_outlier,
            y_outlier=y_outlier,
        )


@if_not_exists(
    "guide",
    "unsupervised",
    "outlier",
    "majority.svg",
)
def gen_majority(path):
    x, y = load_dataset("CBF", progress=False)
    x_outlier, y_outlier = majority_outliers(x, y, n_outliers=0.05)
    original_projection, outlier_projection = make_projection(x=x, x_outlier=x_outlier)

    for fig, ax in yield_and_save_plot(path, figure=mkfigure):
        plot_inlier_outlier(
            ax,
            original_projection=original_projection,
            outlier_projection=outlier_projection,
            x_outlier=x_outlier,
            y_outlier=y_outlier,
        )


@if_not_exists(
    "guide",
    "unsupervised",
    "outlier",
    "kmeans.svg",
)
def gen_kmeans(path):
    x, y = load_dataset("CBF", progress=False)
    x_outlier, y_outlier = kmeans_outliers(x, n_outliers=0.05)
    original_projection, outlier_projection = make_projection(x=x, x_outlier=x_outlier)

    for fig, ax in yield_and_save_plot(path, figure=mkfigure):
        plot_inlier_outlier(
            ax,
            original_projection=original_projection,
            outlier_projection=outlier_projection,
            x_outlier=x_outlier,
            y_outlier=y_outlier,
        )


@if_not_exists(
    "guide",
    "unsupervised",
    "outlier",
    "emmott.svg",
)
def gen_emmott(path):
    x, y = load_dataset("CBF", progress=False)
    x_outlier, y_outlier = emmott_outliers(
        x,
        y,
        n_outliers=0.05,
        scale=4,
        difficulty=1,
        variation="tight",
        transform=None,
        random_state=123,
    )
    original_projection, outlier_projection = make_projection(x=x, x_outlier=x_outlier)

    for fig, ax in yield_and_save_plot(path, figure=mkfigure):
        plot_inlier_outlier(
            ax,
            original_projection=original_projection,
            outlier_projection=outlier_projection,
            x_outlier=x_outlier,
            y_outlier=y_outlier,
        )


@if_not_exists(
    "guide",
    "unsupervised",
    "outlier",
    "emmott-hard.svg",
)
def gen_emmott_hard(path):
    x, y = load_dataset("CBF", progress=False)
    x_outlier, y_outlier = emmott_outliers(
        x,
        y,
        n_outliers=0.05,
        scale=4,
        difficulty=4,
        variation="dispersed",
        transform=None,
        random_state=123,
    )
    original_projection, outlier_projection = make_projection(x=x, x_outlier=x_outlier)

    for fig, ax in yield_and_save_plot(path, figure=mkfigure):
        plot_inlier_outlier(
            ax,
            original_projection=original_projection,
            outlier_projection=outlier_projection,
            x_outlier=x_outlier,
            y_outlier=y_outlier,
        )
