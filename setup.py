# -*- coding: utf-8 -*-

import os

from setuptools import setup
from setuptools.extension import Extension

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


def _merge_build_options(name, options, build_args):
    for arg, value in build_args.items():
        if arg in options:
            options[arg].extend(value)
        else:
            options[arg] = value

    return options


if __name__ == "__main__":
    from Cython.Build import cythonize

    build_type = os.getenv("WILDBOAR_BUILD", "default")
    build_args = BUILD_ARGS.get(build_type)

    if build_args is None:
        raise RuntimeError("%s is not a valid build type" % build_type)

    if os.getenv("WILDBOAR_BUILD_WERROR", 1):
        build_args["extra_compile_args"].append("-Werror")

    if os.name == "posix":
        build_args["libraries"].append("m")

    extensions = {
        "wildboar.utils.misc": {
            "sources": ["src/wildboar/utils/misc.pyx"],
        },
        "wildboar.distance._distance": {
            "sources": ["src/wildboar/distance/_distance.pyx"],
        },
        "wildboar.distance._metric": {
            "sources": ["src/wildboar/distance/_metric.pyx"],
        },
        "wildboar.distance._elastic": {
            "sources": ["src/wildboar/distance/_elastic.pyx"],
        },
        "wildboar.distance._mass": {
            "sources": ["src/wildboar/distance/_mass.pyx"],
        },
        "wildboar.distance._matrix_profile": {
            "sources": ["src/wildboar/distance/_matrix_profile.pyx"],
        },
        "wildboar.tree._ctree": {
            "sources": ["src/wildboar/tree/_ctree.pyx"],
        },
        "wildboar.tree._cptree": {
            "sources": ["src/wildboar/tree/_cptree.pyx"],
        },
        "wildboar.transform._feature": {
            "sources": ["src/wildboar/transform/_feature.pyx"],
        },
        "wildboar.transform._cshapelet": {
            "sources": ["src/wildboar/transform/_cshapelet.pyx"],
        },
        "wildboar.transform._crocket": {
            "sources": ["src/wildboar/transform/_crocket.pyx"],
        },
        "wildboar.transform._cinterval": {
            "sources": ["src/wildboar/transform/_cinterval.pyx"],
        },
        "wildboar.transform._cpivot": {
            "sources": ["src/wildboar/transform/_cpivot.pyx"],
        },
        "wildboar.transform._cfeature_transform": {
            "sources": ["src/wildboar/transform/_cfeature_transform.pyx"],
        },
        "wildboar.transform.catch22._catch22": {
            "sources": [
                "src/wildboar/transform/catch22/_catch22.pyx",
                "src/wildboar/transform/catch22/src/catch22.c",
            ],
            "include_dirs": ["src/wildboar/transform/catch22/src/"],
        },
        "wildboar.utils._fft._pocketfft": {
            "sources": [
                "src/wildboar/utils/_fft/_pocketfft.pyx",
                "src/wildboar/utils/_fft/src/pocketfft.c",
            ],
            "include_dirs": ["src/wildboar/utils/_fft/src/"],
        },
        "wildboar.utils.stats": {
            "sources": ["src/wildboar/utils/stats.pyx"],
        },
        "wildboar.utils.rand": {
            "sources": ["src/wildboar/utils/rand.pyx"],
        },
        "wildboar.utils.data": {
            "sources": ["src/wildboar/utils/data.pyx"],
        },
        "wildboar.utils.parallel": {
            "sources": ["src/wildboar/utils/parallel.pyx"],
        },
    }
    ext_modules = [
        Extension(name, **_merge_build_options(name, options, build_args))
        for name, options in extensions.items()
    ]

    setup(ext_modules=cythonize(ext_modules))
