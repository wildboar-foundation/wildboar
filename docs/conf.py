# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import pathlib
import sys

SEM_VER_REGEX = "^v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"

from pkg_resources import get_distribution, parse_version

release = get_distribution("wildboar").version
version = ".".join(release.split(".")[:3])
# -- Project information -----------------------------------------------------

project = "Wildboar"
description = "Time series learning with Python"
copyright = "2020, Isak Samsten"
author = "Isak Samsten"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "sphinx_multiversion",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autoapi_dirs = "autoapi"
autoapi_root = "."
autoapi_template_dir = "_templates/autoapi/"
autoapi_ignore = ["*tests*", "_*.py"]
autoapi_keep_files = True
autoapi_add_toctree_entry = False


autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_theme_options = {}
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
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
smv_branch_whitelist = r"^master|\d+\.\d+$"
# smv_remote_whitelist = r"^origin$"
smv_released_pattern = r"^refs/(heads|remotes)/\d+\.\d+$"
smv_tag_whitelist = None


def setup(app):
    def read_latest_version(app, config):
        import pathlib
        import re
        from sphinx_multiversion import git

        gitroot = pathlib.Path(
            git.get_toplevel_path(cwd=os.path.abspath(app.confdir))
        ).resolve()
        gitrefs = list(git.get_all_refs(gitroot))
        latest_version_tags = {}
        for ver, metadata in config.smv_metadata.items():
            latest_version_tags[ver] = "dev"
            current_version = re.match("(\d)\.(\d)(?:.X)?", ver)
            if current_version:
                matching_tags = []
                for gitref in gitrefs:
                    match = re.match(SEM_VER_REGEX, gitref.name)
                    if gitref.source == "tags" and match:
                        if match.group(1) == current_version.group(1) and match.group(
                            2
                        ) == current_version.group(2):
                            matching_tags.append(gitref.name.replace("v", ""))
                matching_tags.sort(key=lambda x: parse_version(x))
                latest_version_tags[ver] = matching_tags[-1]
                metadata["version"] = matching_tags[-1]
            else:
                metadata["version"] = ver

        if config.smv_current_version != "master":
            config.version = config.smv_metadata[config.smv_current_version]["version"]
            config.release = config.version

        config.smv_latest_version_tags = latest_version_tags
        config.smv_current_version_tag = latest_version_tags[config.smv_current_version]
        version_sorted = sorted(
            config.smv_metadata.keys(), key=lambda x: parse_version(x)
        )
        config.smv_latest_stable = version_sorted[-1]

    def set_latest_version(app, pagename, templatename, context, doctree):
        from sphinx_multiversion.sphinx import VersionInfo

        versioninfo = VersionInfo(
            app, context, app.config.smv_metadata, app.config.smv_current_version
        )
        context["latest_version_tags"] = app.config.smv_latest_version_tags
        context["latest_stable_version"] = versioninfo[app.config.smv_latest_stable]
        context["latest_develop_version"] = versioninfo["master"]

    app.connect("config-inited", read_latest_version)
    app.connect("html-page-context", set_latest_version)
