# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
from pkg_resources import parse_version, get_distribution

# sys.path.insert(0, os.path.abspath("_extensions"))

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
    "sphinx_simpleversion",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "autoapi.extension",
    "sphinx_design",
    "nbsphinx",
    "sphinx_copybutton",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

versions_develop_branch = "master"

autoapi_dirs = ["../src/"]
autoapi_root = "."
autoapi_template_dir = "_templates/autoapi/"
autoapi_ignore = ["*tests*", "_*.py"]
autoapi_keep_files = True
autoapi_add_toctree_entry = False
autoapi_python_class_content = "both"
autoapi_member_order = "groupwise"

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
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

current_branch_name = get_current_branch()


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

    return "https://github.com/isaksamsten/wildboar/blob/%s/src/%s" % (
        current_branch_name,
        filename,
    )
