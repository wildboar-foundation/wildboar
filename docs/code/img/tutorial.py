import matplotlib.pylab as plt
from wildboar.datasets import load_synthetic_control


def time_series_predict():
    x, y = load_synthetic_control()
    fig, ax = plt.subplots()
    for i in [-1, 12]:
        ax.plot(x[i, :], label="%s" % str(y[i]))
    ax.legend()
    return fig


PLOT_DICT = {"time_series_predict": time_series_predict}
