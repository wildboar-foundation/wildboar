# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

from logging import log
import os

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
    "sphinx_panels",
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
smv_branch_whitelist = r"^master|\d+\.\d+\.X$"
smv_remote_whitelist = r"^origin$"
smv_released_pattern = r"^refs/(heads|remotes)/\d+\.\d+\.X$"
smv_tag_whitelist = None


def setup(app):
    import re
    import pathlib
    from sphinx.util.logging import getLogger

    logger = getLogger(__name__)

    def is_tag_for(gitref, major, minor):
        match = re.match(SEM_VER_REGEX, gitref.name)
        return (
            gitref.source == "tags"
            and match
            and match.group(1) == major
            and match.group(2) == minor
        )

    def read_latest_version(app, config):
        if not hasattr(config, "smv_metadata") or len(config.smv_metadata) == 0:
            return

        logger.info("[CONF] installed version %s", config.release)

        from sphinx_multiversion import git

        gitroot = pathlib.Path(
            git.get_toplevel_path(cwd=os.path.abspath(app.confdir))
        ).resolve()
        gitrefs = list(git.get_all_refs(gitroot))
        latest_version_tags = {}
        for ver, metadata in config.smv_metadata.items():
            current_version = re.match("(\d)\.(\d)(?:.X)?", ver)
            if current_version:
                major = current_version.group(1)
                minor = current_version.group(2)
                matching_tags = [
                    re.sub("^v", "", gitref.name)
                    for gitref in gitrefs
                    if is_tag_for(gitref, major, minor)
                ]
                matching_tags.sort(key=parse_version)
                latest_tag = matching_tags[-1] if matching_tags else ver
                latest_version_tags[ver] = latest_tag
            elif ver == "master":
                latest_version_tags[ver] = config.version
            else:
                logger.warning("[CONF] using branch version for tag (%s)", ver)
                latest_version_tags[ver] = ver

            logger.info(
                "[CONF] latest version tag for '%s' is '%s'",
                ver,
                latest_version_tags[ver],
            )
            metadata["version"] = latest_version_tags[ver]

        if config.smv_current_version != "master":
            config.version = config.smv_metadata[config.smv_current_version]["version"]
            config.release = config.version

        config.smv_latest_version_tags = latest_version_tags
        config.smv_current_version_tag = latest_version_tags[config.smv_current_version]

        version_sorted = sorted(
            config.smv_metadata.keys(),
            key=lambda x: parse_version(re.sub(".X$", "", x)),
        )
        config.smv_latest_stable = version_sorted[-1] if version_sorted else "master"

        logger.info("[DOCS] latest stable version is %s", config.smv_latest_stable)

    def set_latest_version(app, pagename, templatename, context, doctree):
        if not hasattr(app.config, "smv_metadata") or len(app.config.smv_metadata) == 0:
            return

        from sphinx_multiversion.sphinx import VersionInfo

        versioninfo = VersionInfo(
            app, context, app.config.smv_metadata, app.config.smv_current_version
        )
        context["current_version_tag"] = app.config.smv_current_version_tag
        context["latest_version_tags"] = app.config.smv_latest_version_tags
        context["latest_stable_version"] = versioninfo[app.config.smv_latest_stable]

    app.connect("config-inited", read_latest_version)
    app.connect("html-page-context", set_latest_version)
