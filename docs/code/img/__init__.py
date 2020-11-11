import os
from .tutorial import PLOT_DICT as TUTORIAL


def build_all(output_path, plot_dict, output_ext=".png"):
    output_path = os.path.join(output_path, "tutorial")
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for name, plotter in plot_dict.items():
        fig = plotter()
        fig.savefig(os.path.join(output_path, name + output_ext))
