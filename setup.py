# -*- coding: utf-8 -*-

import os

from setuptools import setup
from setuptools.extension import Extension


# Lazy loading
class np_include_dirs(str):
    def __str__(self):
        import numpy as np

        return np.get_include()


BUILD_TYPE = os.getenv("WILDBOAR_BUILD") or "default"

BUILD_ARGS = {
    "default": {
        "extra_compile_args": ["-O2"],
        "extra_link_args": [],
        "libraries": [],
    },
    "optimized": {
        "extra_compile_args": [
            "-O3",
            "-march=native",
            "-msse",
            "-msse2",
            "-mfma",
            "-mfpmath=sse",
        ],
        "extra_link_args": [],
        "libraries": [],
    },
}

DEFINE_MACRO_NUMPY_C_API = [
    (
        "NPY_NO_DEPRECATED_API",
        "NPY_1_7_API_VERSION",
    )
]


def _merge_build_options(options, build_args):
    for arg, value in build_args.items():
        if arg in options:
            options[arg].extend(value)
        else:
            options[arg] = value

    return options


if __name__ == "__main__":
    from Cython.Build import cythonize

    def include_dirs(*args):
        return [np_include_dirs(), *args]

    build_args = BUILD_ARGS.get(BUILD_TYPE)
    if build_args is None:
        raise RuntimeError("%s is not a valid build type" % BUILD_TYPE)

    if os.name == "unix":
        build_args["libraries"].append("m")

    extensions = {
        "wildboar.utils.misc": {
            "sources": ["src/wildboar/utils/misc.pyx"],
            "include_dirs": include_dirs(),
        },
        "wildboar.distance._distance": {
            "sources": ["src/wildboar/distance/_distance.pyx"],
            "include_dirs": include_dirs(),
        },
        "wildboar.distance._metric": {
            "sources": ["src/wildboar/distance/_metric.pyx"],
            "include_dirs": include_dirs(),
            "define_macros": DEFINE_MACRO_NUMPY_C_API,
        },
        "wildboar.distance._elastic": {
            "sources": ["src/wildboar/distance/_elastic.pyx"],
            "include_dirs": include_dirs(),
        },
        "wildboar.distance._mass": {
            "sources": ["src/wildboar/distance/_mass.pyx"],
            "include_dirs": include_dirs(),
        },
        "wildboar.distance._matrix_profile": {
            "sources": ["src/wildboar/distance/_matrix_profile.pyx"],
            "include_dirs": include_dirs(),
        },
        "wildboar.tree._ctree": {
            "sources": ["src/wildboar/tree/_ctree.pyx"],
            "include_dirs": include_dirs(),
        },
        "wildboar.tree._cptree": {
            "sources": ["src/wildboar/tree/_cptree.pyx"],
            "include_dirs": include_dirs(),
        },
        "wildboar.transform._feature": {
            "sources": ["src/wildboar/transform/_feature.pyx"],
            "include_dirs": include_dirs(),
        },
        "wildboar.transform._cshapelet": {
            "sources": ["src/wildboar/transform/_cshapelet.pyx"],
            "include_dirs": include_dirs(),
        },
        "wildboar.transform._crocket": {
            "sources": ["src/wildboar/transform/_crocket.pyx"],
            "include_dirs": include_dirs(),
        },
        "wildboar.transform._cinterval": {
            "sources": ["src/wildboar/transform/_cinterval.pyx"],
            "include_dirs": include_dirs(),
        },
        "wildboar.transform._cpivot": {
            "sources": ["src/wildboar/transform/_cpivot.pyx"],
            "include_dirs": include_dirs(),
        },
        "wildboar.transform._cfeature_transform": {
            "sources": ["src/wildboar/transform/_cfeature_transform.pyx"],
            "include_dirs": include_dirs(),
        },
        "wildboar.transform.catch22._catch22": {
            "sources": [
                "src/wildboar/transform/catch22/_catch22.pyx",
                "src/wildboar/transform/catch22/src/catch22.c",
            ],
            "include_dirs": include_dirs("src/wildboar/transform/catch22/src/"),
        },
        "wildboar.utils._fft._pocketfft": {
            "sources": [
                "src/wildboar/utils/_fft/_pocketfft.pyx",
                "src/wildboar/utils/_fft/src/pocketfft.c",
            ],
            "include_dirs": include_dirs("src/wildboar/utils/_fft/src/"),
        },
        "wildboar.utils.stats": {
            "sources": ["src/wildboar/utils/stats.pyx"],
            "include_dirs": include_dirs(),
        },
        "wildboar.utils.rand": {
            "sources": ["src/wildboar/utils/rand.pyx"],
            "include_dirs": include_dirs(),
        },
        "wildboar.utils.data": {
            "sources": ["src/wildboar/utils/data.pyx"],
            "include_dirs": include_dirs(),
        },
        "wildboar.utils.parallel": {
            "sources": ["src/wildboar/utils/parallel.pyx"],
            "include_dirs": include_dirs(),
        },
    }
    ext_modules = [
        Extension(name, **_merge_build_options(options, build_args))
        for name, options in extensions.items()
    ]

    setup(ext_modules=cythonize(ext_modules))
