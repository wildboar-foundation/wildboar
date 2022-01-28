# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wildboar is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
# General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
import numpy as np

from wildboar.utils import check_array

try:
    import matplotlib.pylab as plt
    from matplotlib.cm import get_cmap
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
except ModuleNotFoundError as e:
    from wildboar.utils import DependencyMissing

    matplotlib_missing = DependencyMissing(e, package="matplotlib")
    plt = matplotlib_missing
    get_cmap = matplotlib_missing
    Normalize = matplotlib_missing
    LineCollection = matplotlib_missing
    Line2D = matplotlib_missing


class MidpointNormalize(Normalize):
    """Normalise the colorbar."""

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        super().__init__(vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def plot_time_domain(
    x,
    *,
    y=None,
    ax=None,
    alpha=0.5,
    linewidth=0.5,
    zorder=-1,
    cmap="Dark2",
):
    """Plot the samples in the time domain

    Parameters
    ----------
    x : array-like of shape (n_sample, n_timestep)
        The samples
    y : array-like of shape (n_samples, ), optional
        The labels, by default None
    ax : Axes, optional
        The matplotlib Axes-object, by default None
    cmap : str, optional
        The colormap, by default "Dark2"
    """
    if ax is None:
        fig, ax = plt.subplots()

    x = check_array(np.atleast_2d(x), allow_multivariate=False, contiguous=False)
    if y is not None:
        y = check_array(y, ensure_2d=False, allow_nd=False, contiguous=False)
        if y.shape[0] != x.shape[0]:
            raise ValueError("The number of labels and samples are not the same.")

        label, idx, inv = np.unique(y, return_inverse=True, return_index=True)
        cmap = get_cmap(cmap, len(label))
    else:
        cmap = get_cmap(cmap, 1)
        inv = np.zeros(x.shape[0])

    x_axis = np.arange(x.shape[-1] + 1)
    collection = LineCollection(
        [list(zip(x_axis, x[i])) for i in range(x.shape[0])],
        colors=[cmap(inv[i]) for i in range(x.shape[0])],
        zorder=zorder,
        linewidth=linewidth,
        alpha=alpha,
    )
    ax.add_collection(collection)
    if y is not None:
        ax.legend([Line2D([0], [0], color=cmap(inv[i])) for i in idx], label)

    ax.set_xlim([0, x.shape[-1] - 1])
    ax.set_ylim([np.min(x) - np.std(x), np.max(x) + np.std(x)])

    return ax


def plot_frequency_domain(
    x,
    *,
    y=None,
    ax=None,
    jitter=False,
    sample_spacing=1,
    frequency=False,
    cmap="Dark2",
):
    """Plot the samples in the freqency domain

    Parameters
    ----------
    x : array-like of shape (n_sample, n_timestep)
        The samples
    y : array-like of shape (n_samples, ), optional
        The labels, by default None
    ax : Axes, optional
        The matplotlib Axes-object, by default None
    jitter : bool, optional
        Add jitter to the amplitude lines, by default False
    sample_spacing : int, optional
        The frequency domain sample spacing, by default 1
    frequency : bool, optional
        Show the frequency bins, by default False
    cmap : str, optional
        The colormap, by default "Dark2"
    """
    if ax is None:
        fig, ax = plt.subplots()

    x = check_array(x, allow_multivariate=False, contiguous=False)
    if y is not None:
        y = check_array(1, ensure_2d=False, allow_nd=False, contiguous=False)
        if y.shape[0] != x.shape[0]:
            raise ValueError("The number of samples and lables are not the same.")

        label, idx, inv = np.unique(y, return_inverse=True, return_index=True)
        cmap = get_cmap(cmap, len(label))
    else:
        cmap = get_cmap(cmap, 1)
        inv = np.zeros(x.shape[0])

    n_freqs = int(x.shape[-1] // 2)
    x_freq = np.abs(np.fft.fft(x, axis=1)[:, 1 : n_freqs + 1]) / n_freqs
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
            color=cmap(idx[i]),
            alpha=0.3,
            linewidth=1,
            zorder=-1,
        )

    if y is not None:
        ax.legend([Line2D([0], [0], color=cmap(inv[i])) for i in idx], label)

    ticks = ax.get_xticks().astype(int)[: len(x_axis)]
    ticks[0] = 1
    ax.set_xticks(ticks)
    x_label = np.fft.fftfreq(x.shape[-1], d=sample_spacing)[ticks]
    ax.set_xticklabels("%.2f" % lab for lab in x_label)
    ax.set_xlim([0.5, n_freqs + 0.5])
    ax.set_ylim(0, max_freq)

    return ax
