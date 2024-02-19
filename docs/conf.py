# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import pkgutil
import pathlib
import sys

from importlib import metadata
from sphinx.util.logging import getLogger

import wildboar

sys.path.insert(0, os.path.abspath("_gen_figures"))
sys.path.insert(0, os.path.abspath(".") + "/_extensions")

logger = getLogger(__name__)

release = wildboar.__version__
version = release

if ".dev" in version:
    version_match = "dev"
else:
    version_match = ".".join(version.split(".")[:2])

logger.info("Building documentation for %s, matching %s", version, version_match)
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
    "autoapi.extension",
    "lightdarkimg",
    "execute",
    "numpydoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_inline_tabs",
    "sphinx_toggleprompt",
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

pygments_style = "github-light-colorblind"
pygments_dark_style = "github-dark-colorblind"
syntax_highlight = "short"
add_function_parentheses = False

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
    # "undoc-members",
    # "show-inheritance",
    "show-module-summary",
    # "special-members",
    "imported-members",
    "inherited-members",
]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "logo": {
        "text": "Wildboar",
        "image_light": "logo.png",
        "image_dark": "logo.png",
    },
    "show_version_warning_banner": True,
    "navbar_start": ["navbar-logo", "version-switcher"],
    "switcher": {
        "json_url": "https://wildboar.dev/_versions.json",
        "version_match": version_match,
    },
    "check_switcher": False,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/wildboar-foundation/wildboar/",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/wildboar/",
            "icon": "fa-custom fa-pypi",
        },
    ],
    "navigation_with_keys": False,
}

html_context = {
    "default_mode": "auto",
}

html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
    "css/pygments-override.css",
]
html_js_files = ["js/fa.js"]

rst_prolog = """
.. role:: python(code)
   :language: python
   :class: highlight
"""

execute_pre_code = """
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(threshold=30)
"""

execute_light_dark = True


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
        "master" if ".dev" in version else "v%s" % version,
        filename,
    )


regenerate_plots = os.environ.get("REGENERATE_PLOTS", False)

basepath = pathlib.Path("_static", "fig")
for modinfo in pkgutil.iter_modules(["_gen_figures"]):
    if modinfo.name.startswith("gen"):
        logger.info(f"Generating figures for: '{modinfo.name}'")
        module = modinfo.module_finder.find_module(modinfo.name).load_module(
            modinfo.name
        )
        funcs = list(module.__dict__.values())
        for func in funcs:
            if callable(func) and func.__name__.startswith("gen_"):
                logger.info(f" - '{func.__name__}'")
                func(basepath, force=regenerate_plots)
