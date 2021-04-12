import os

from .counterfactual import PLOT_DICT as COUNTERFACTUALS
from .tutorial import PLOT_DICT as TUTORIAL

__all__ = ["COUNTERFACTUALS", "TUTORIAL"]


def build_all(output_path, plot_dict, subdir=None, output_ext=".png"):
    if subdir:
        output_path = os.path.join(output_path, subdir)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for name, plotter in plot_dict.items():
        print("Plotting: %s" % name)
        fig = plotter()
        fig.savefig(os.path.join(output_path, name + output_ext))
