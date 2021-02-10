# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages
from setuptools.extension import Extension


# Lazy loading
class np_include_dirs(str):
    def __str__(self):
        import numpy as np

        return np.get_include()


def read(rel_path):
    import codecs

    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


def declare_extension(
    root_module,
    ext_file,
    compile_args=None,
    link_args=None,
    use_openmp=False,
    include_dirs=None,
    libraries=None,
    extra_compile_args=None,
    extra_link_args=None,
):
    if not isinstance(root_module, list):
        root_module = [root_module]

    ext_module = "%s.%s" % (".".join(root_module), ext_file)
    ext_file = ["%s.pyx" % os.path.join(*root_module, ext_file)]

    compile_args = compile_args or []
    extra_compile_args = extra_compile_args or []
    compile_args.extend(extra_compile_args)

    link_args = link_args or []
    extra_link_args = extra_link_args or []
    link_args.extend(extra_link_args)
    libraries = libraries or []
    if use_openmp:
        compile_args.insert(0, "-fopenmp")
        link_args.insert(0, "-fopenmp")

    import platform

    if platform.system() != "Windows":
        libraries.insert(0, "m")

    return Extension(
        ext_module,
        ext_file,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=include_dirs,
        libraries=libraries,
    )


def declare_extensions(ext_module, ext_files, **kwargs):
    return [declare_extension(ext_module, ext_file, **kwargs) for ext_file in ext_files]


PACKAGE_NAME = "wildboar"

SHORT_DESCRIPTION = "Time series learning with Python"

with open(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md"),
    encoding="utf-8",
) as f:
    DESCRIPTION = f.read()

BUILD_TYPE = os.getenv("WILDBOAR_BUILD") or "default"

BUILD_ARGS = {
    "default": {
        "compile_args": ["-O2"],
        "link_args": [],
        "libraries": [],
    },
    "optimized": {
        "compile_args": [
            "-O2",
            "-march=native",
            "-msse",
            "-msse2",
            "-mfma",
            "-mfpmath=sse",
        ],
        "link_args": [],
        "libraries": [],
    },
}

include_dirs = [np_include_dirs()]

build_args = BUILD_ARGS.get(BUILD_TYPE)
if build_args is None:
    raise RuntimeError("%s is not a valid build type" % BUILD_TYPE)

wildboar_ext = declare_extensions(
    "wildboar", ["_utils"], use_openmp=False, include_dirs=include_dirs, **build_args
)

distance_ext = declare_extensions(
    ["wildboar", "distance"],
    [
        "_distance",
        "_euclidean_distance",
        "_dtw_distance",
    ],
    use_openmp=False,
    include_dirs=include_dirs,
    **build_args
)

tree_ext = declare_extensions(
    ["wildboar", "tree"],
    [
        "_impurity",
        "_tree_builder",
    ],
    use_openmp=False,
    include_dirs=include_dirs,
    **build_args
)

cython_ext_modules = wildboar_ext + distance_ext + tree_ext
setup(
    name="wildboar",
    version=get_version("wildboar/__init__.py"),
    author="Isak Samsten",
    author_email="isak@samsten.se",
    url="https://github.com/isakkarlsson/wildboar",
    description=SHORT_DESCRIPTION,
    long_description=DESCRIPTION,
    long_description_content_type="text/markdown",
    license="GPLv3",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Programming Language :: Cython",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
    ],
    setup_requires=[
        "cython>=0.29.14",
        "numpy>=1.17.4",
        "setuptools>=18.0",
    ],
    install_requires=[
        "numpy>=1.17.4",
        "scikit-learn>=0.21.3",
        "scipy>=1.3.2",
    ],
    python_requires=">=3.7.0",
    provides=["wildboar"],
    keywords=["machine learning", "time series distance"],
    ext_modules=cython_ext_modules,
    packages=find_packages(),
    package_data={
        "wildboar": ["*.pxd", "*.pyx"],
        "wildboar.distance": ["*.pxd"],
        "wildboar.tree": ["*.pxd"],
    },
    zip_safe=False,
)
