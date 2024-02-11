# Copied and modified from Matplotlib
import ast
import contextlib
import doctest
import itertools
import os
import re
import shutil
import sys
import textwrap
import traceback
from io import StringIO
from os.path import relpath
from pathlib import Path

import jinja2
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
from docutils.parsers.rst import Directive, directives
from docutils.parsers.rst.directives.images import Image
from matplotlib import _pylab_helpers, cbook
from matplotlib.backend_bases import FigureManagerBase
from matplotlib.colors import ListedColormap
from sphinx.errors import ExtensionError

_WBCOLORMAP = ListedColormap(
    [
        [0.700151, 0.002745, 0.700612],
        [0.700191, 0.010833, 0.697186],
        [0.700226, 0.019196, 0.693784],
        [0.700255, 0.027497, 0.690410],
        [0.700279, 0.036129, 0.687067],
        [0.700299, 0.044535, 0.683750],
        [0.700315, 0.052201, 0.680467],
        [0.700328, 0.059479, 0.677234],
        [0.700339, 0.066138, 0.674024],
        [0.700347, 0.072500, 0.670865],
        [0.700355, 0.078557, 0.667747],
        [0.700362, 0.084489, 0.664665],
        [0.700369, 0.090118, 0.661647],
        [0.700378, 0.095602, 0.658662],
        [0.700389, 0.100919, 0.655740],
        [0.700403, 0.106180, 0.652854],
        [0.700422, 0.111272, 0.650035],
        [0.700447, 0.116281, 0.647260],
        [0.700479, 0.121141, 0.644532],
        [0.700521, 0.126029, 0.641864],
        [0.700575, 0.130794, 0.639246],
        [0.700642, 0.135455, 0.636684],
        [0.700726, 0.140079, 0.634169],
        [0.700829, 0.144685, 0.631701],
        [0.700953, 0.149193, 0.629284],
        [0.701102, 0.153641, 0.626923],
        [0.701279, 0.158084, 0.624615],
        [0.701486, 0.162466, 0.622349],
        [0.701729, 0.166801, 0.620147],
        [0.702012, 0.171076, 0.617979],
        [0.702337, 0.175336, 0.615867],
        [0.702707, 0.179555, 0.613800],
        [0.703122, 0.183741, 0.611790],
        [0.703582, 0.187936, 0.609823],
        [0.704093, 0.192033, 0.607911],
        [0.704658, 0.196154, 0.606029],
        [0.705275, 0.200200, 0.604213],
        [0.705946, 0.204298, 0.602424],
        [0.706671, 0.208310, 0.600685],
        [0.707439, 0.212332, 0.598992],
        [0.708265, 0.216333, 0.597337],
        [0.709140, 0.220302, 0.595714],
        [0.710053, 0.224219, 0.594134],
        [0.711015, 0.228186, 0.592595],
        [0.712011, 0.232099, 0.591073],
        [0.713044, 0.235997, 0.589585],
        [0.714113, 0.239864, 0.588131],
        [0.715214, 0.243717, 0.586706],
        [0.716339, 0.247586, 0.585309],
        [0.717485, 0.251421, 0.583920],
        [0.718642, 0.255233, 0.582557],
        [0.719825, 0.259028, 0.581216],
        [0.721012, 0.262798, 0.579897],
        [0.722212, 0.266557, 0.578580],
        [0.723423, 0.270315, 0.577284],
        [0.724641, 0.274034, 0.576009],
        [0.725860, 0.277747, 0.574723],
        [0.727075, 0.281433, 0.573468],
        [0.728294, 0.285107, 0.572216],
        [0.729516, 0.288786, 0.570967],
        [0.730731, 0.292428, 0.569729],
        [0.731948, 0.296063, 0.568502],
        [0.733159, 0.299689, 0.567287],
        [0.734368, 0.303280, 0.566067],
        [0.735568, 0.306883, 0.564853],
        [0.736769, 0.310471, 0.563661],
        [0.737963, 0.314024, 0.562464],
        [0.739164, 0.317574, 0.561261],
        [0.740347, 0.321115, 0.560077],
        [0.741527, 0.324632, 0.558885],
        [0.742714, 0.328153, 0.557711],
        [0.743886, 0.331654, 0.556532],
        [0.745056, 0.335166, 0.555357],
        [0.746225, 0.338633, 0.554196],
        [0.747393, 0.342111, 0.553026],
        [0.748552, 0.345565, 0.551874],
        [0.749704, 0.349024, 0.550714],
        [0.750868, 0.352465, 0.549561],
        [0.752015, 0.355908, 0.548416],
        [0.753158, 0.359324, 0.547260],
        [0.754311, 0.362743, 0.546124],
        [0.755451, 0.366162, 0.544977],
        [0.756589, 0.369556, 0.543837],
        [0.757726, 0.372949, 0.542705],
        [0.758860, 0.376350, 0.541561],
        [0.759990, 0.379722, 0.540437],
        [0.761122, 0.383103, 0.539305],
        [0.762246, 0.386459, 0.538189],
        [0.763371, 0.389823, 0.537065],
        [0.764497, 0.393183, 0.535933],
        [0.765621, 0.396533, 0.534816],
        [0.766735, 0.399878, 0.533691],
        [0.767850, 0.403228, 0.532586],
        [0.768965, 0.406563, 0.531468],
        [0.770080, 0.409898, 0.530357],
        [0.771197, 0.413225, 0.529252],
        [0.772302, 0.416545, 0.528136],
        [0.773408, 0.419864, 0.527036],
        [0.774514, 0.423171, 0.525928],
        [0.775620, 0.426487, 0.524822],
        [0.776727, 0.429800, 0.523723],
        [0.777826, 0.433105, 0.522621],
        [0.778921, 0.436413, 0.521521],
        [0.780016, 0.439713, 0.520427],
        [0.781110, 0.443016, 0.519342],
        [0.782202, 0.446308, 0.518256],
        [0.783285, 0.449611, 0.517171],
        [0.784373, 0.452911, 0.516077],
        [0.785450, 0.456202, 0.515007],
        [0.786526, 0.459505, 0.513939],
        [0.787590, 0.462792, 0.512865],
        [0.788657, 0.466096, 0.511800],
        [0.789713, 0.469411, 0.510744],
        [0.790763, 0.472703, 0.509696],
        [0.791807, 0.476015, 0.508661],
        [0.792845, 0.479325, 0.507622],
        [0.793877, 0.482635, 0.506589],
        [0.794892, 0.485949, 0.505565],
        [0.795900, 0.489268, 0.504572],
        [0.796901, 0.492594, 0.503560],
        [0.797894, 0.495924, 0.502571],
        [0.798879, 0.499261, 0.501599],
        [0.799853, 0.502601, 0.500622],
        [0.800811, 0.505942, 0.499654],
        [0.801763, 0.509301, 0.498715],
        [0.802706, 0.512652, 0.497778],
        [0.803641, 0.516005, 0.496832],
        [0.804568, 0.519378, 0.495907],
        [0.805481, 0.522745, 0.495000],
        [0.806395, 0.526120, 0.494092],
        [0.807296, 0.529496, 0.493183],
        [0.808190, 0.532873, 0.492294],
        [0.809081, 0.536254, 0.491419],
        [0.809960, 0.539633, 0.490532],
        [0.810839, 0.543029, 0.489643],
        [0.811716, 0.546423, 0.488777],
        [0.812583, 0.549811, 0.487916],
        [0.813449, 0.553199, 0.487046],
        [0.814314, 0.556601, 0.486178],
        [0.815178, 0.560003, 0.485315],
        [0.816042, 0.563405, 0.484456],
        [0.816898, 0.566806, 0.483600],
        [0.817752, 0.570200, 0.482748],
        [0.818608, 0.573611, 0.481898],
        [0.819464, 0.577019, 0.481035],
        [0.820320, 0.580426, 0.480189],
        [0.821177, 0.583832, 0.479337],
        [0.822036, 0.587241, 0.478479],
        [0.822888, 0.590654, 0.477626],
        [0.823741, 0.594071, 0.476774],
        [0.824596, 0.597489, 0.475924],
        [0.825452, 0.600903, 0.475075],
        [0.826310, 0.604330, 0.474213],
        [0.827167, 0.607752, 0.473350],
        [0.828020, 0.611173, 0.472504],
        [0.828883, 0.614590, 0.471655],
        [0.829740, 0.618022, 0.470796],
        [0.830598, 0.621451, 0.469943],
        [0.831459, 0.624886, 0.469087],
        [0.832315, 0.628325, 0.468218],
        [0.833174, 0.631759, 0.467368],
        [0.834038, 0.635197, 0.466506],
        [0.834897, 0.638641, 0.465638],
        [0.835759, 0.642088, 0.464790],
        [0.836622, 0.645532, 0.463931],
        [0.837488, 0.648990, 0.463055],
        [0.838348, 0.652439, 0.462201],
        [0.839214, 0.655902, 0.461342],
        [0.840083, 0.659357, 0.460469],
        [0.840947, 0.662826, 0.459614],
        [0.841815, 0.666286, 0.458749],
        [0.842684, 0.669764, 0.457878],
        [0.843548, 0.673236, 0.457012],
        [0.844421, 0.676714, 0.456145],
        [0.845291, 0.680187, 0.455282],
        [0.846160, 0.683675, 0.454415],
        [0.847033, 0.687166, 0.453536],
        [0.847907, 0.690654, 0.452675],
        [0.848784, 0.694149, 0.451800],
        [0.849662, 0.697652, 0.450924],
        [0.850534, 0.701158, 0.450049],
        [0.851410, 0.704667, 0.449178],
        [0.852296, 0.708180, 0.448296],
        [0.853170, 0.711699, 0.447413],
        [0.854049, 0.715220, 0.446541],
        [0.854936, 0.718742, 0.445655],
        [0.855818, 0.722269, 0.444788],
        [0.856699, 0.725811, 0.443905],
        [0.857584, 0.729347, 0.443021],
        [0.858470, 0.732893, 0.442142],
        [0.859359, 0.736436, 0.441247],
        [0.860250, 0.739997, 0.440361],
        [0.861136, 0.743552, 0.439474],
        [0.862032, 0.747115, 0.438584],
        [0.862924, 0.750684, 0.437706],
        [0.863818, 0.754257, 0.436810],
        [0.864708, 0.757833, 0.435919],
        [0.865604, 0.761413, 0.435028],
        [0.866502, 0.765006, 0.434120],
        [0.867400, 0.768593, 0.433228],
        [0.868307, 0.772195, 0.432343],
        [0.869206, 0.775798, 0.431443],
        [0.870103, 0.779408, 0.430538],
        [0.871012, 0.783023, 0.429635],
        [0.871923, 0.786651, 0.428733],
        [0.872831, 0.790273, 0.427834],
        [0.873742, 0.793911, 0.426929],
        [0.874658, 0.797545, 0.426015],
        [0.875582, 0.801192, 0.425119],
        [0.876504, 0.804847, 0.424205],
        [0.877438, 0.808510, 0.423294],
        [0.878381, 0.812177, 0.422386],
        [0.879330, 0.815859, 0.421480],
        [0.880295, 0.819541, 0.420578],
        [0.881278, 0.823243, 0.419683],
        [0.882276, 0.826956, 0.418775],
        [0.883302, 0.830680, 0.417880],
        [0.884356, 0.834420, 0.416995],
        [0.885442, 0.838174, 0.416101],
        [0.886573, 0.841950, 0.415222],
        [0.887744, 0.845741, 0.414352],
        [0.888981, 0.849563, 0.413497],
        [0.890275, 0.853401, 0.412658],
        [0.891647, 0.857274, 0.411817],
        [0.893098, 0.861176, 0.411008],
        [0.894649, 0.865109, 0.410221],
        [0.896296, 0.869084, 0.409443],
        [0.898055, 0.873088, 0.408695],
        [0.899941, 0.877132, 0.407976],
        [0.901952, 0.881224, 0.407286],
        [0.904103, 0.885355, 0.406631],
        [0.906404, 0.889529, 0.406013],
        [0.908850, 0.893747, 0.405420],
    ],
    name="wildboar",
)
try:
    matplotlib.colormaps.register(_WBCOLORMAP)
except:
    pass


def light_dark_paths(path: Path):
    file = Path(path.parts[-1])
    path = Path(*path.parts[:-1])
    ext = file.suffix
    name = file.stem

    return path.joinpath(f"{name}-light{ext}"), path.joinpath(f"{name}-dark{ext}")


def interleave(b):
    o = b[::2]
    if len(b) % 2 == 0:
        start = -1
    else:
        start = -2
    r = b[start::-2]
    oj = 0
    rj = 0
    new = []
    for i in range(len(b)):
        if i % 2 == 0:
            new.append(o[oj])
            oj += 1
        else:
            new.append(r[rj])
            rj += 1
    return new


colors = interleave(
    [
        matplotlib.colors.rgb2hex(rgb)
        for rgb in matplotlib.colormaps["wildboar"].resampled(10).colors
    ]
)


matplotlib.use("agg")

__version__ = 2


# -----------------------------------------------------------------------------
# Registration hook
# -----------------------------------------------------------------------------


def _option_boolean(arg):
    if not arg or not arg.strip():
        # no argument given, assume used as a flag
        return True
    elif arg.strip().lower() in ("no", "0", "false"):
        return False
    elif arg.strip().lower() in ("yes", "1", "true"):
        return True
    else:
        raise ValueError(f"{arg!r} unknown boolean")


def _option_context(arg):
    if arg in [None, "reset", "close-figs"]:
        return arg
    raise ValueError("Argument should be None or 'reset' or 'close-figs'")


def _option_format(arg):
    return directives.choice(arg, ("python", "doctest"))


def _option_output_format(arg):
    if arg is None:
        return "plain"
    return directives.choice(arg, ("code", "plain"))


def _option_card_width(arg):
    return directives.choice(arg, ("auto", "25%", "50%", "75%", "100%"))


def _option_carousel(arg):
    return directives.choice(arg, (1, 2, 3))


def mark_plot_labels(app, document):
    """
    To make plots referenceable, we need to move the reference from the
    "htmlonly" (or "latexonly") node to the actual figure node itself.
    """
    for name, explicit in document.nametypes.items():
        if not explicit:
            continue
        labelid = document.nameids[name]
        if labelid is None:
            continue
        node = document.ids[labelid]
        if node.tagname in ("html_only", "latex_only"):
            for n in node:
                if n.tagname == "figure":
                    sectname = name
                    for c in n:
                        if c.tagname == "caption":
                            sectname = c.astext()
                            break

                    node["ids"].remove(labelid)
                    node["names"].remove(name)
                    n["ids"].append(labelid)
                    n["names"].append(name)
                    document.settings.env.labels[name] = (
                        document.settings.env.docname,
                        labelid,
                        sectname,
                    )
                    break


class ExecuteDirective(Directive):
    """The ``.. execute::`` directive, as documented in the module's docstring."""

    has_content = True
    required_arguments = 0
    optional_arguments = 2
    final_argument_whitespace = False
    option_spec = {
        "alt": directives.unchanged,
        "height": directives.length_or_unitless,
        "width": directives.length_or_percentage_or_unitless,
        "scale": directives.nonnegative_int,
        "align": Image.align,
        "class": directives.class_option,
        "include-source": _option_boolean,
        "show-return": _option_boolean,
        "show-output": _option_output_format,
        "show-output-label": _option_boolean,
        "show-return-label": _option_boolean,
        "show-source-link": _option_boolean,
        "card": _option_boolean,
        "carousel": _option_carousel,
        "card-width": _option_card_width,
        "format": _option_format,
        "context": _option_context,
        "nofigs": directives.flag,
        "caption": directives.unchanged,
        "link-text": directives.unchanged,
    }

    def run(self):
        """Run the plot directive."""
        try:
            rcParams = {
                "image.cmap": "wildboar",
                "axes.prop_cycle": cycler(color=colors),
            }
            light_dark = (
                self.state_machine.document.settings.env.config.execute_light_dark
            )

            if light_dark:
                light, is_doctest, build_dir_link, source_file_name, errors_light = run(
                    arguments=self.arguments,
                    content=self.content,
                    options=self.options,
                    rc_params=rcParams,
                    dark_mode=False,
                    state_machine=self.state_machine,
                    state=self.state,
                    lineno=self.lineno,
                )
                if is_doctest:
                    raise ValueError("doctest syntax is not supported")

                dark, _, _, _, errors_dark = run(
                    arguments=self.arguments,
                    content=self.content,
                    options=self.options,
                    rc_params=rcParams,
                    dark_mode=True,
                    state_machine=self.state_machine,
                    state=self.state,
                    lineno=self.lineno,
                )

                if "align" not in self.options:
                    self.options["align"] = "center"

                opts = [
                    f":{key}: {val}"
                    for key, val in self.options.items()
                    if key in ("alt", "height", "width", "scale", "align")
                ]
                light_opts = opts + [":class: only-light, card-image"]
                dark_opts = opts + [":class: only-dark, card-image"]
                if "card" in self.options and not self.options["card"]:
                    template = TEMPLATE
                else:
                    template = CARD_TEMPLATE

                total_lines = []
                for (source_code, (kind, output), stdout, src_name, light_images), (
                    _,
                    _,
                    _,
                    _,
                    dark_images,
                ) in zip(light, dark):
                    result = jinja2.Template(template).render(
                        default_fmt="png",
                        images=list(zip(light_images, dark_images)),
                        build_dir=build_dir_link,
                        source_code=source_code,
                        src_name=src_name,
                        carousel=self.options.get("carousel", 2),
                        card_width=self.options.get("card-width", "auto"),
                        show_return=self.options.get("show-return", False),
                        show_return_label=self.options.get("show-return-label", True),
                        show_output_label=self.options.get("show-output-label", False),
                        output_format=self.options.get("show-output", None),
                        return_output=output,
                        return_kind=kind,
                        stdout=stdout,
                        link_text=self.options.get("link-text", "Download source"),
                        light_options=light_opts,
                        dark_options=dark_opts,
                    )
                    total_lines.extend(result.split("\n"))
            else:
                light, is_doctest, build_dir_link, source_file_name, errors_light = run(
                    arguments=self.arguments,
                    content=self.content,
                    options=self.options,
                    rc_params=rcParams,
                    dark_mode=False,
                    state_machine=self.state_machine,
                    state=self.state,
                    lineno=self.lineno,
                )
                if is_doctest:
                    raise ValueError("doctest syntax is not supported")

                if "align" not in self.options:
                    self.options["align"] = "center"

                opts = [
                    f":{key}: {val}"
                    for key, val in self.options.items()
                    if key in ("alt", "height", "width", "scale", "align")
                ]
                light_opts = opts + [":class: card-image"]
                dark_opts = None
                if "card" in self.options and not self.options["card"]:
                    template = TEMPLATE
                else:
                    template = CARD_TEMPLATE

                total_lines = []
                for source_code, (
                    kind,
                    output,
                ), stdout, src_name, light_images in light:
                    result = jinja2.Template(template).render(
                        default_fmt="png",
                        images=list(zip(light_images, light_images)),
                        build_dir=build_dir_link,
                        source_code=source_code,
                        src_name=src_name,
                        carousel=self.options.get("carousel", 2),
                        card_width=self.options.get("card-width", "auto"),
                        show_return=self.options.get("show-return", False),
                        show_return_label=self.options.get("show-return-label", True),
                        show_output_label=self.options.get("show-output-label", False),
                        output_format=self.options.get("show-output", None),
                        return_output=output,
                        return_kind=kind,
                        stdout=stdout,
                        link_text=self.options.get("link-text", "Download source"),
                        light_options=light_opts,
                        dark_options=dark_opts,
                    )
                    total_lines.extend(result.split("\n"))

            if total_lines:
                self.state_machine.insert_input(total_lines, source=source_file_name)

            return errors_light
        except Exception as e:
            raise self.error(str(e))


def _copy_css_file(app, exc):
    if exc is None and app.builder.format == "html":
        src = cbook._get_data_path("plot_directive/plot_directive.css")
        dst = app.outdir / Path("_static")
        dst.mkdir(exist_ok=True)
        # Use copyfile because we do not want to copy src's permissions.
        shutil.copyfile(src, dst / Path("plot_directive.css"))


# -----------------------------------------------------------------------------
# Doctest handling
# -----------------------------------------------------------------------------


def contains_doctest(text):
    try:
        # check if it's valid Python as-is
        compile(text, "<string>", "exec")
        return False
    except SyntaxError:
        pass
    r = re.compile(r"^\s*>>>", re.M)
    m = r.search(text)
    return bool(m)


def _split_code_at_show(text, function_name):
    """Split code at plt.show()."""

    is_doctest = contains_doctest(text)
    if function_name is None:
        parts = []
        part = []
        for line in text.split("\n"):
            if (not is_doctest and line.startswith("plt.show(")) or (
                is_doctest and line.strip() == ">>> plt.show()"
            ):
                part.append(line)
                parts.append("\n".join(part))
                part = []
            else:
                part.append(line)
        if "\n".join(part).strip():
            parts.append("\n".join(part))
    else:
        parts = [text]
    return is_doctest, parts


# -----------------------------------------------------------------------------
# Template
# -----------------------------------------------------------------------------

_SOURCECODE = """
{{ source_code }}
"""

TEMPLATE_SRCSET = (
    _SOURCECODE
    + """
   {% for img in images %}
   .. figure-mpl:: {{ build_dir }}/{{ img.basename }}.{{ default_fmt }}
      {% for option in options -%}
      {{ option }}
      {% endfor %}
      {%- if caption -%}
      {{ caption }}  {# appropriate leading whitespace added beforehand #}
      {% endif -%}
      {%- if srcset -%}
        :srcset: {{ build_dir }}/{{ img.basename }}.{{ default_fmt }}
        {%- for sr in srcset -%}
            , {{ build_dir }}/{{ img.basename }}.{{ sr }}.{{ default_fmt }} {{sr}}
        {%- endfor -%}
      {% endif %}

   {% endfor %}

"""
)

TEMPLATE = """
{% if source_code %}
.. code-block:: python
   {{ source_code_class }}

{{ source_code }}
{% endif %}
{% for light, dark in images %}

.. image:: {{ build_dir }}/{{ light.basename }}.{{ default_fmt }}
   {% for option in light_options -%}
   {{ option }}
   {% endfor %}

.. image:: {{ build_dir }}/{{ dark.basename }}.{{ default_fmt }}
   {% for option in dark_options -%}
   {{ option }}
   {% endfor %}
{% endfor %}
{%- if src_name -%}
:download:`Source code <{{ build_dir }}/{{ src_name }}>`
{%- endif -%}
"""

CARD_TEMPLATE = """
{%- if images or source_code or (show_return and return_kind) or stdout %}
.. grid:: 1
    :padding: 0
    :margin: 0
    {% if source_code or (show_return and return_kind) or stdout %}
    .. grid-item::
        :columns: 12
    {% if source_code %}
        .. code-block:: python
{% filter indent(width=8) %}
{{ source_code }}
{%- endfilter %}
    {%- endif %}
    {% if show_return and return_kind %}
        {%- if return_kind == "html" %}
        .. raw:: html

            <div class="sd-text-center", data-mime-type="text/html">
{%- filter indent(width=12) %}
{{ return_output|trim }}
{%- endfilter %}
            </div>
        {%- elif return_kind == "plain" %}
        .. code-block::
            :class: sd-mt-0 {%- if show_return_label %} execute-return {%- endif %}
{% filter indent(width=12) %}
{{ return_output }}
{%- endfilter %}
        {%- endif %}
    {%- endif %}
    {% if output_format == "code" %}
        .. code-block::
            :class: sd-mt-0 {%- if show_output_label %} execute-output {%- endif %}

{% filter indent(width=12) %}
{{ stdout }}
{%- endfilter %}
    {%- endif %}
    {%- endif -%}
{% if output_format == "plain" %}
{{stdout|trim}}
{% if images %}
.. grid:: 1
    :padding: 0
    :margin: 0
{% endif -%}
{% endif -%}
    {% if images %}
    .. grid-item::
        :columns: 12
        {% if images|length > 0 %}
        .. card-carousel:: {{ carousel }}
        {% for light, dark in images %}
            .. card::
                :width: {{ card_width }}
                :shadow: none
                :text-align: center
                :margin: auto

                .. image:: {{ build_dir }}/{{ light.basename }}.{{ default_fmt }}
                    {% for option in light_options -%}
                    {{ option }}
                    {% endfor %}
                {% if light != dark -%}
                .. image:: {{ build_dir }}/{{ dark.basename }}.{{ default_fmt }}
                    {% for option in dark_options -%}
                    {{ option }}
                    {% endfor %}
                {% endif -%}

        {%- endfor %}
        {% else -%}
        {% for light, dark in images %}
        .. card::
            :width: {{ card_width }}
            :shadow: none
            :text-align: center
            :margin: auto

            .. image:: {{ build_dir }}/{{ light.basename }}.{{ default_fmt }}
                {% for option in light_options -%}
                {{ option }}
                {% endfor %}

            .. image:: {{ build_dir }}/{{ dark.basename }}.{{ default_fmt }}
                {% for option in dark_options -%}
                {{ option }}
                {% endfor %}

        {%- endfor %}
        {%- endif %}
    {%- if src_name %}
    .. grid-item::
        :columns: 12
        :child-align: center
        :class: sd-text-center

        :download:`{{ link_text }} <{{ build_dir }}/{{ src_name }}>`
    {%- endif %}
    {%- endif %}
{%- elif src_name -%}
:download:`{{ link_text }} <{{ build_dir }}/{{ src_name }}>`
{%- endif %}

"""
exception_template = """
.. only:: html

   [`source code <%(linkdir)s/%(basename)s.py>`__]

Exception occurred rendering plot.

"""

# the context of the plot for all directives specified with the
# :context: option
plot_context = dict()


class ImageFile:
    def __init__(self, basename, dirname):
        self.basename = basename
        self.dirname = dirname
        self.formats = []

    def filename(self, format):
        return os.path.join(self.dirname, f"{self.basename}.{format}")

    def filenames(self):
        return [self.filename(fmt) for fmt in self.formats]


def out_of_date(original, derived, includes=None):
    """
    Return whether *derived* is out-of-date relative to *original* or any of
    the RST files included in it using the RST include directive (*includes*).
    *derived* and *original* are full paths, and *includes* is optionally a
    list of full paths which may have been included in the *original*.
    """
    if not os.path.exists(derived):
        return True

    if includes is None:
        includes = []
    files_to_check = [original, *includes]

    def out_of_date_one(original, derived_mtime):
        return os.path.exists(original) and derived_mtime < os.stat(original).st_mtime

    derived_mtime = os.stat(derived).st_mtime
    return any(out_of_date_one(f, derived_mtime) for f in files_to_check)


class PlotError(RuntimeError):
    pass


def exec_with_return(code, globals=None, locals=None):
    a = ast.parse(code)
    last_expression = None
    if a.body:
        if isinstance(a_last := a.body[-1], ast.Expr):
            last_expression = ast.unparse(a.body.pop())
        elif isinstance(a_last, ast.Assign):
            last_expression = ast.unparse(a_last.targets[0])
        elif isinstance(a_last, (ast.AnnAssign, ast.AugAssign)):
            last_expression = ast.unparse(a_last.target)
    exec(ast.unparse(a), globals, locals)
    if last_expression:
        return eval(last_expression, globals, locals)


def _run_code(code, code_path, ns=None, function_name=None):
    """
    Import a Python module from a path, and run the function given by
    name, if function_name is not None.
    """

    # Change the working directory to the directory of the example, so
    # it can get at its data files, if any.  Add its path to sys.path
    # so it can import any helper modules sitting beside it.
    pwd = os.getcwd()
    if setup.config.execute_working_directory is not None:
        try:
            os.chdir(setup.config.execute_working_directory)
        except OSError as err:
            raise OSError(
                f"{err}\n`execute_working_directory` option in "
                f"Sphinx configuration file must be a valid "
                f"directory path"
            ) from err
        except TypeError as err:
            raise TypeError(
                f"{err}\n`execute_working_directory` option in "
                f"Sphinx configuration file must be a string or "
                f"None"
            ) from err
    elif code_path is not None:
        dirname = os.path.abspath(os.path.dirname(code_path))
        os.chdir(dirname)

    return_value = None
    stdout = None
    with StringIO() as buf, cbook._setattr_cm(
        sys, argv=[code_path], path=[os.getcwd(), *sys.path]
    ), contextlib.redirect_stdout(buf):
        try:
            if ns is None:
                ns = {}
            if not ns:
                if setup.config.execute_pre_code is None:
                    exec(
                        "import numpy as np\n" "from matplotlib import pyplot as plt\n",
                        ns,
                    )
                else:
                    exec(str(setup.config.execute_pre_code), ns)
            if "__main__" in code:
                ns["__name__"] = "__main__"

            # Patch out non-interactive show() to avoid triggering a warning.
            with cbook._setattr_cm(FigureManagerBase, show=lambda self: None):
                return_value = exec_with_return(code, ns)
                if function_name is not None:
                    return_value = eval(function_name + "()", ns)

            stdout = buf.getvalue()
        except (Exception, SystemExit) as err:
            raise PlotError(traceback.format_exc()) from err
        finally:
            os.chdir(pwd)

    return ns, return_value, stdout


def clear_state(execute_rcparams, *, close=True):
    if close:
        plt.close("all")
    matplotlib.rcParams.update(execute_rcparams)


def get_plot_formats(config):
    default_dpi = {"png": 80, "hires.png": 200, "pdf": 200}
    formats = []
    execute_formats = config.execute_formats
    for fmt in execute_formats:
        if isinstance(fmt, str):
            if ":" in fmt:
                suffix, dpi = fmt.split(":")
                formats.append((str(suffix), int(dpi)))
            else:
                formats.append((fmt, default_dpi.get(fmt, 80)))
        elif isinstance(fmt, (tuple, list)) and len(fmt) == 2:
            formats.append((str(fmt[0]), int(fmt[1])))
        else:
            raise PlotError('invalid image format "%r" in execute_formats' % fmt)
    return formats


def _parse_srcset(entries):
    """
    Parse srcset for multiples...
    """
    srcset = {}
    for entry in entries:
        entry = entry.strip()
        if len(entry) >= 2:
            mult = entry[:-1]
            srcset[float(mult)] = entry
        else:
            raise ExtensionError(f"srcset argument {entry!r} is invalid.")
    return srcset


def render_figures(
    code,
    code_path,
    output_dir,
    output_base,
    context,
    function_name,
    config,
    dark_mode,
    rc_params={},
    context_reset=False,
    close_figs=False,
    code_includes=None,
):
    """
    Run a pyplot script and save the images in *output_dir*.

    Save the images under *output_dir* with file names derived from
    *output_base*
    """

    if function_name is not None:
        output_base = f"{output_base}_{function_name}"
    formats = get_plot_formats(config)

    # Try to determine if all images already exist

    is_doctest, code_pieces = _split_code_at_show(code, function_name)
    # Look for single-figure output files first
    img = ImageFile(output_base, output_dir)
    for format, dpi in formats:
        if context or out_of_date(
            code_path, img.filename(format), includes=code_includes
        ):
            all_exists = False
            break
        img.formats.append(format)
    else:
        all_exists = True

    if all_exists:
        return [(code, [img])]

    # Then look for multi-figure output files
    results = []
    for i, code_piece in enumerate(code_pieces):
        images = []
        for j in itertools.count():
            if len(code_pieces) > 1:
                img = ImageFile("%s_%02d_%02d" % (output_base, i, j), output_dir)
            else:
                img = ImageFile("%s_%02d" % (output_base, j), output_dir)
            for fmt, dpi in formats:
                if context or out_of_date(
                    code_path, img.filename(fmt), includes=code_includes
                ):
                    all_exists = False
                    break
                img.formats.append(fmt)

            # assume that if we have one, we have them all
            if not all_exists:
                all_exists = j > 0
                break
            images.append(img)
        if not all_exists:
            break
        results.append((code_piece, images))
    else:
        all_exists = True

    if all_exists:
        return results

    # We didn't find the files, so build them

    results = []
    ns = plot_context if context else {}

    if context_reset:
        clear_state(rc_params)
        plot_context.clear()

    close_figs = not context or close_figs

    for i, code_piece in enumerate(code_pieces):
        if dark_mode:
            matplotlib.style.use("dark_background")
        else:
            matplotlib.style.use("default")

        if not context or config.execute_apply_rcparams:
            clear_state(rc_params, close=close_figs)
        elif close_figs:
            plt.close("all")

        _ns, return_value, stdout = _run_code(
            doctest.script_from_examples(code_piece) if is_doctest else code_piece,
            code_path,
            ns,
            function_name,
        )

        return_repr = (None, None)
        if return_value is not None:
            if hasattr(return_value, "_repr_html_"):
                return_repr = ("html", return_value._repr_html_())
            else:
                return_repr = ("plain", return_value.__repr__())

        images = []
        fig_managers = _pylab_helpers.Gcf.get_all_fig_managers()
        for j, figman in enumerate(fig_managers):
            if len(fig_managers) == 1 and len(code_pieces) == 1:
                img = ImageFile(output_base, output_dir)
            elif len(code_pieces) == 1:
                img = ImageFile("%s_%02d" % (output_base, j), output_dir)
            else:
                img = ImageFile("%s_%02d_%02d" % (output_base, i, j), output_dir)
            images.append(img)

            for fmt, dpi in formats:
                try:
                    figman.canvas.figure.savefig(
                        img.filename(fmt), dpi=dpi, transparent=True
                    )
                    if fmt == formats[0][0] and config.execute_srcset:
                        # save a 2x, 3x etc version of the default...
                        srcset = _parse_srcset(config.execute_srcset)
                        for mult, suffix in srcset.items():
                            fm = f"{suffix}.{fmt}"
                            img.formats.append(fm)
                            figman.canvas.figure.savefig(
                                img.filename(fm), dpi=int(dpi * mult)
                            )
                except Exception as err:
                    raise PlotError(traceback.format_exc()) from err
                img.formats.append(fmt)

        results.append((code_piece, return_repr, stdout, images))

    if not context or config.execute_apply_rcparams:
        clear_state(rc_params, close=not context)

    return results


def run(
    *, arguments, content, options, rc_params, dark_mode, state_machine, state, lineno
):
    document = state_machine.document
    config = document.settings.env.config
    nofigs = "nofigs" in options

    if config.execute_srcset and setup.app.builder.name == "singlehtml":
        raise ExtensionError(
            "execute_srcset option not compatible with single HTML writer"
        )

    formats = get_plot_formats(config)
    default_fmt = formats[0][0]

    options.setdefault("include-source", config.execute_include_source)
    options.setdefault("show-source-link", config.execute_html_show_source_link)

    if "class" in options:
        # classes are parsed into a list of string, and output by simply
        # printing the list, abusing the fact that RST guarantees to strip
        # non-conforming characters
        options["class"] = ["plot-directive"] + options["class"]
    else:
        options.setdefault("class", ["plot-directive"])
    keep_context = "context" in options
    context_opt = None if not keep_context else options["context"]
    if context_opt is None:
        context_opt = config.execute_context

    rst_file = document.attributes["source"]
    rst_dir = os.path.dirname(rst_file)

    if len(arguments):
        if not config.execute_basedir:
            source_file_name = os.path.join(
                setup.app.builder.srcdir, directives.uri(arguments[0])
            )
        else:
            source_file_name = os.path.join(
                setup.confdir, config.execute_basedir, directives.uri(arguments[0])
            )
        # If there is content, it will be passed as a caption.
        caption = "\n".join(content)

        # Enforce unambiguous use of captions.
        if "caption" in options:
            if caption:
                raise ValueError(
                    "Caption specified in both content and options."
                    " Please remove ambiguity."
                )
            # Use caption option
            caption = options["caption"]

        # If the optional function name is provided, use it
        if len(arguments) == 2:
            function_name = arguments[1]
        else:
            function_name = None

        code = Path(source_file_name).read_text(encoding="utf-8")
        output_base = os.path.basename(source_file_name)
    else:
        source_file_name = rst_file
        code = textwrap.dedent("\n".join(map(str, content)))
        counter = document.attributes.get("_plot_counter", 0) + 1
        document.attributes["_plot_counter"] = counter
        base, ext = os.path.splitext(os.path.basename(source_file_name))
        output_base = "%s-%d.py" % (base, counter)
        function_name = None
        caption = options.get("caption", "")

    base, source_ext = os.path.splitext(output_base)
    if source_ext in (".py", ".rst", ".txt"):
        output_base = base
    else:
        source_ext = ""

    # ensure that LaTeX includegraphics doesn't choke in foo.bar.pdf filenames
    output_base = output_base.replace(".", "-")

    # is it in doctest format?
    is_doctest = contains_doctest(code)
    if "format" in options:
        if options["format"] == "python":
            is_doctest = False
        else:
            is_doctest = True

    # determine output directory name fragment
    source_rel_name = relpath(source_file_name, setup.confdir)
    source_rel_dir = os.path.dirname(source_rel_name).lstrip(os.path.sep)

    # build_dir: where to place output files (temporarily)
    build_dir = os.path.join(
        os.path.dirname(setup.app.doctreedir), "plot_directive", source_rel_dir
    )
    # get rid of .. in paths, also changes pathsep
    # see note in Python docs for warning about symbolic links on Windows.
    # need to compare source and dest paths at end
    build_dir = os.path.normpath(build_dir)
    os.makedirs(build_dir, exist_ok=True)

    # how to link to files from the RST file
    try:
        build_dir_link = relpath(build_dir, rst_dir).replace(os.path.sep, "/")
    except ValueError:
        # on Windows, relpath raises ValueError when path and start are on
        # different mounts/drives
        build_dir_link = build_dir

    # get list of included rst files so that the output is updated when any
    # plots in the included files change. These attributes are modified by the
    # include directive (see the docutils.parsers.rst.directives.misc module).
    try:
        source_file_includes = [
            os.path.join(os.getcwd(), t[0]) for t in state.document.include_log
        ]
    except AttributeError:
        # the document.include_log attribute only exists in docutils >=0.17,
        # before that we need to inspect the state machine
        possible_sources = {
            os.path.join(setup.confdir, t[0]) for t in state_machine.input_lines.items
        }
        source_file_includes = [f for f in possible_sources if os.path.isfile(f)]
    # remove the source file itself from the includes
    try:
        source_file_includes.remove(source_file_name)
    except ValueError:
        pass

    # save script (if necessary)
    if options["show-source-link"]:
        Path(build_dir, output_base + source_ext).write_text(
            doctest.script_from_examples(code)
            if source_file_name == rst_file and is_doctest
            else code,
            encoding="utf-8",
        )

    # make figures
    try:
        results = render_figures(
            code=code,
            code_path=source_file_name,
            output_dir=build_dir,
            output_base=output_base,
            context=keep_context,
            function_name=function_name,
            config=config,
            rc_params=rc_params,
            dark_mode=dark_mode,
            context_reset=context_opt == "reset",
            close_figs=context_opt == "close-figs",
            code_includes=source_file_includes,
        )
        errors = []
    except PlotError as err:
        reporter = state.memo.reporter
        sm = reporter.system_message(
            2,
            "Exception occurred in plotting {}\n from {}:\n{}".format(
                output_base, source_file_name, err
            ),
            line=lineno,
        )
        results = [(code, None, None, [])]
        errors = [sm]

    # Properly indent the caption
    if caption and config.execute_srcset:
        caption = f":caption: {caption}"
    elif caption:
        caption = "\n" + "\n".join(
            "      " + line.strip() for line in caption.split("\n")
        )
    # generate output restructuredtext
    total_lines = []
    all_images = []
    for j, (code_piece, return_value, stdout, images) in enumerate(results):
        if options["include-source"]:
            if is_doctest:
                lines = code_piece.splitlines()
            else:
                lines = textwrap.indent(code_piece, "    ").splitlines()

            source_code = "\n".join(lines)
        else:
            source_code = None

        if j == 0 and options["show-source-link"]:
            src_name = output_base + source_ext
        else:
            src_name = None

        if nofigs:
            images = []

        all_images.append((source_code, return_value, stdout, src_name, images))

    return all_images, is_doctest, build_dir_link, source_file_name, errors


def setup(app):
    setup.app = app
    setup.config = app.config
    setup.confdir = app.confdir
    app.add_directive("execute", ExecuteDirective)
    app.add_config_value("execute_light_dark", True, True)
    app.add_config_value("execute_pre_code", None, True)
    app.add_config_value("execute_include_source", True, True)
    app.add_config_value("execute_html_show_source_link", False, True)
    app.add_config_value("execute_formats", ["png"], True)
    app.add_config_value("execute_basedir", None, True)
    app.add_config_value("execute_html_show_formats", True, True)
    app.add_config_value("execute_rcparams", {}, True)
    app.add_config_value("execute_apply_rcparams", True, True)
    app.add_config_value("execute_working_directory", None, True)
    app.add_config_value("execute_template", None, True)
    app.add_config_value("execute_srcset", [], True)
    app.add_config_value("execute_context", "close-figs", True)
    app.connect("doctree-read", mark_plot_labels)
    app.connect("build-finished", _copy_css_file)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
