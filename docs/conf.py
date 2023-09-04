# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os

from pkg_resources import get_distribution, parse_version
from sphinx_simpleversion import get_current_branch

current_release = parse_version(get_distribution("wildboar").version)
release = current_release.public
version = f"{current_release.major}.{current_release.minor}.{current_release.micro}"

# -- Project information -----------------------------------------------------

project = "Wildboar"
description = "Time series learning with Python"
copyright = "2023, Isak Samsten"
author = "Isak Samsten"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "matplotlib.sphinxext.plot_directive",
    "sphinx_simpleversion",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinx.ext.linkcode",
    "autoapi.extension",
    "sphinx_design",
    "sphinx_copybutton",
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "wildboar": ("https://wildboar.dev/main", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

pygments_style = "github-light-colorblind"
pygments_dark_style = "github-dark-colorblind"
syntax_highlight = "short"
add_function_parentheses = False

# TODO: Enable once we have fewer warnings
# numpydoc_validation_checks = {
#     "all",
#     "SA01",
#     "EX01",
# }

versions_develop_branch = "master"

autoapi_dirs = ["../src/"]
autoapi_root = "api"
autoapi_template_dir = "_templates/autoapi/"
autoapi_ignore = ["*tests*", "_*.py"]
autoapi_keep_files = True
autoapi_add_toctree_entry = True
autoapi_python_class_content = "class"
autoapi_member_order = "groupwise"

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    # "special-members",
    "imported-members",
]

html_theme = "furo"
html_theme_options = {
    "light_logo": "logo.png",
    "dark_logo": "logo.png",
}
html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "versions.html",
        "sidebar/scroll-end.html",
    ],
}

html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]
current_branch_name = get_current_branch()

rst_prolog = """
.. role:: python(code)
   :language: python
   :class: highlight
"""


# Find the source file given a module
def find_source(info):
    import importlib
    import inspect

    obj = importlib.import_module(info["module"])
    for part in info["fullname"].split("."):
        obj = getattr(obj, part)

    fn = os.path.normpath(inspect.getsourcefile(obj))
    fn_split = fn.split(os.sep)
    fn_index = fn_split.index("wildboar")
    fn = os.path.join(*fn_split[fn_index:])
    source, lineno = inspect.getsourcelines(obj)
    return fn, lineno, lineno + len(source) - 1


def linkcode_resolve(domain, info):
    if domain != "py" or not info["module"]:
        return None
    try:
        filename = "%s#L%d-L%d" % find_source(info)
    except:
        filename = info["module"].replace(".", "/") + ".py"

    return "https://github.com/wildboar-foundation/wildboar/blob/%s/src/%s" % (
        current_branch_name,
        filename,
    )
