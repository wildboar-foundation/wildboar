import os
import matplotlib.pylab as plt


def time_series_predict():
    from wildboar.datasets import load_synthetic_control
    x, y = load_synthetic_control()
    fig, ax = plt.subplots()
    for i in [-1, 12]:
        ax.plot(x[i, :], label="%s" % str(y[i]))
    ax.legend()
    return fig


_plotters = {
    'time_series_predict': time_series_predict
}


def build_all(output_path, output_ext=".png"):
    output_path = os.path.join(output_path, "tutorial")
    for name, plotter in _plotters.items():
        fig = plotter()
        fig.savefig(os.path.join(output_path, name + output_ext))
