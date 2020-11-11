# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(1, os.path.abspath('code/'))
from wildboar import __version__

# -- Project information -----------------------------------------------------

project = 'Wildboar'
description = "Time series learning with Python"
copyright = '2020, Isak Samsten'
author = 'Isak Samsten'

# The full version, including alpha/beta/rc tags
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'autoapi.sphinx',
    'sphinx_multiversion',
    'sphinx.ext.autodoc',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.napoleon'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autoclass_content = 'both'
autodoc_default_flags = [
    "members",
    "inherited-members",
    "show-inheritance",
]

autoapi_modules = {'wildboar': {'prune': True}}
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'
html_theme_options = {
    'github_user': 'isaksamsten',
    'github_repo': 'wildboar',
    'github_button': True,
    'github_type': 'star',
    'fixed_sidebar': True,
    'sidebar_collapse': True,
    'show_relbar_bottom': True,
}
html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "relations.html",
        "searchbox.html",
        "versions.html",
    ],
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
img_path = os.path.abspath("_static/img")


# Build figures before documents are read.
def build_fig_handler(app, env, docnames):
    if not os.path.exists(img_path):
        os.mkdir(img_path)

    import img
    img.build_all(img_path, img.TUTORIAL)


def setup(app):
    app.connect('env-before-read-docs', build_fig_handler)
