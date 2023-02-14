# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
import subprocess
import re

from pkg_resources import parse_version
from setuptools_scm import get_version
from sphinx.util.logging import getLogger

sys.path.insert(0, os.path.abspath("sphinxext"))

from version import SimpleVersion, find_version_by_name

logger = getLogger(__name__)

release = get_version("..")
VERSION = parse_version(release)
version = f"{VERSION.major}.{VERSION.minor}.{VERSION.micro}"

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

nbsphinx_prompt_width = "0px"


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
        f"{VERSION.major}.{VERSION.minor}.X" if not VERSION.is_devrelease else "master",
        filename,
    )


def is_tag_version(tag):
    # Excluding major version of 0
    SEMVER = (
        r"^v?([1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
        r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
    )
    return re.match(SEMVER, tag)


def get_versions_from_git():
    with subprocess.Popen(["git tag"], stdout=subprocess.PIPE, shell=True) as cmd:
        tags, _ = cmd.communicate()
        tags = [tag.strip() for tag in tags.decode().splitlines()]
        logger.info(tags)
        return [SimpleVersion(tag) for tag in tags if is_tag_version(tag)]


def get_latest_version_major_minor():
    versions = {}

    for version in get_versions_from_git():
        major_minor = f"{version.version.major}.{version.version.minor}"
        if major_minor not in versions:
            versions[major_minor] = version
        else:
            old_version = versions[major_minor]
            if version > old_version:
                versions[major_minor] = version

    return [value for _, value in sorted(versions.items(), reverse=True)]


versions = get_latest_version_major_minor()
latest_stable_version = versions[0]
develop_version = SimpleVersion("master", dev_version=VERSION)

# Render the development version last
versions.insert(0, develop_version)

logger.info(versions)
html_context = {
    "versions": versions,
    "stable_version": latest_stable_version,
    "develop_version": develop_version,
    "current_version": (
        find_version_by_name(version, versions)
        if not VERSION.is_devrelease
        else develop_version
    ),
}
logger.info(html_context)
