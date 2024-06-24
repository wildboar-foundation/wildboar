import numpy as np
from light_dark import mk_light_dark
from matplotlib.pyplot import get_cmap
from wildboar import iseos
from wildboar.datasets import load_dataset, load_gun_point
from wildboar.datasets.preprocess import minmax_scale, truncate


@mk_light_dark("guide", "datasets", "repository", "preprocess", "minmax_scale.svg")
def gen_preprocess_minmax_scale(fig, ax):
    x, y = load_gun_point()
    ax.plot(x[0])
    x = minmax_scale(x)
    ax.plot(x[0])
    fig.legend(["Original", "Scaled"])


def plot_dim_len(x, fig, ax):
    eos = iseos(x).argmax(axis=2)
    cmap = get_cmap(lut=eos.shape[1])
    for dim in range(eos.shape[1]):
        eos[eos[:, dim] == 0] = x.shape[-1]  # if eos == n_timestep
        ax[dim].scatter(
            np.arange(eos.shape[0]),
            eos[:, dim],
            label="dim %d" % dim,
            marker="x",
            color=cmap(dim),
        )
        ax[dim].set_ylabel(f"dim {dim}")
    fig.tight_layout()


@mk_light_dark(
    "guide", "datasets", "repository", "preprocess", "no-truncate.svg", nrows=3
)
def gen_prerocess_no_truncate(fig, ax):
    x, y = load_dataset(
        "SpokenArabicDigits", repository="wildboar/ucrmts", progress=False
    )
    x = x[0:25, :3, :]
    plot_dim_len(x, fig, ax)


@mk_light_dark("guide", "datasets", "repository", "preprocess", "truncate.svg", nrows=3)
def gen_prerocess_truncate(fig, ax):
    x, y = load_dataset(
        "SpokenArabicDigits", repository="wildboar/ucrmts", progress=False
    )
    x = x[0:25, :3, :]
    plot_dim_len(truncate(x), fig, ax)
