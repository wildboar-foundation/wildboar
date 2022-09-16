# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

SEM_VER_REGEX = r"^v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"  # noqa E501

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
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "autoapi.extension",
    "sphinx_multiversion",
    "sphinx_design",
    "nbsphinx",
    "sphinx_copybutton",
]

if os.getenv("LOCAL_BUILD", 0):
    LOCAL_EXTENSIONS_REMOVE = [
        "sphinx_multiversion",
    ]
    for value in LOCAL_EXTENSIONS_REMOVE:
        extensions.remove(value)

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

if os.getenv("LOCAL_BUILD", 0):
    autoapi_dirs = ["../src/"]
else:
    autoapi_dirs = "autoapi"

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
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
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
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
smv_branch_whitelist = r"^master|\d+\.\d+\.X$"
smv_remote_whitelist = r"^origin$"
# smv_released_pattern = r"^refs/(heads|remotes)/\d+\.\d+\.X$"
smv_tag_whitelist = None

nbsphinx_prompt_width = "0px"


def setup(app):
    import importlib
    import inspect
    import pathlib
    import re

    from sphinx.util.logging import getLogger

    logger = getLogger(__name__)

    def is_tag_for(gitref, major, minor):
        p = parse_version(gitref.name)
        return gitref.source == "tags" and p.major == major and p.minor == minor

    # Build the local version of wildboar into a temporary directory
    if os.getenv("LOCAL_BUILD", 0):

        def build_local_version(app):
            pass

    else:

        def build_local_version(app):
            import subprocess

            logger.info("[CONF] building and installing local version")
            env = os.environ.copy()
            env["SETUPTOOLS_SCM_PRETEND_VERSION"] = "99.9.99"
            version_file = os.path.join(app.srcdir, "../src/wildboar/version.py")
            with open(os.path.abspath(version_file), "w") as f:
                f.write("version='99.9.99'")  # dummy version

            # Build and install local version in temporary directory
            output = subprocess.run(
                ["python", "-m", "pip", "install", "--target", "../_build", "."],
                cwd=os.path.abspath(os.path.join(app.srcdir, "..")),
                env=env,
            )
            output.check_returncode()  # Abort if build failed

    # Find the source file given a module
    def find_source(info):
        obj = importlib.import_module(info["module"])
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)

        fn = os.path.normpath(inspect.getsourcefile(obj))
        fn_split = fn.split(os.sep)
        fn_index = fn_split.index("wildboar")
        fn = os.path.join(*fn_split[fn_index:])
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    def read_latest_version(app, config):
        build_local_version(app)

        def linkcode_resolve(domain, info):
            sys.path.insert(0, "../_build/")  # Insert the local build dir first in path
            from wildboar import __version__

            if (
                not os.getenv("LOCAL_BUILD", 0) and __version__ != "99.9.99"
            ):  # ensure we are using the local build
                raise ValueError("Local build failed")

            if domain != "py" or not info["module"]:
                return None
            try:
                filename = "%s#L%d-L%d" % find_source(info)
            except:
                filename = info["module"].replace(".", "/") + ".py"

            del sys.path[0]
            return "https://github.com/isaksamsten/wildboar/blob/%s/src/%s" % (
                config.smv_current_version
                if not os.getenv("LOCAL_BUILD", 0)
                else "master",
                filename,
            )

        config.linkcode_resolve = linkcode_resolve
        if not os.getenv("LOCAL_BUILD", 0):
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
                current_version = re.match(r"(\d)\.(\d)(?:.X)?", ver)
                if current_version:
                    major = int(current_version.group(1))
                    minor = int(current_version.group(2))
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
                config.version = config.smv_metadata[config.smv_current_version][
                    "version"
                ]
                config.release = config.version

            config.smv_latest_version_tags = latest_version_tags
            config.smv_current_version_tag = latest_version_tags[
                config.smv_current_version
            ]

            version_sorted = sorted(
                config.smv_metadata.keys(),
                key=lambda x: parse_version(re.sub(".X$", "", x)),
            )
            # TODO: exclude pre-releases?
            config.smv_latest_stable = (
                version_sorted[-1] if version_sorted else "master"
            )

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
