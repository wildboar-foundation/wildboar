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
            "-O2",
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

if __name__ == "__main__":
    from Cython.Build import cythonize

    include_dirs = [np_include_dirs()]

    build_args = BUILD_ARGS.get(BUILD_TYPE)
    if build_args is None:
        raise RuntimeError("%s is not a valid build type" % BUILD_TYPE)

    extensions = [
        (
            "wildboar._utils",
            ["src/wildboar/_utils.pyx"],
            [],
        ),
        (
            "wildboar.distance._distance",
            ["src/wildboar/distance/_distance.pyx"],
            [],
        ),
        (
            "wildboar.distance._euclidean_distance",
            ["src/wildboar/distance/_euclidean_distance.pyx"],
            [],
        ),
        (
            "wildboar.distance._dtw_distance",
            ["src/wildboar/distance/_dtw_distance.pyx"],
            [],
        ),
        (
            "wildboar.tree._tree_builder",
            ["src/wildboar/tree/_tree_builder.pyx"],
            [],
        ),
        (
            "wildboar._data",
            ["src/wildboar/_data.pyx"],
            [],
        ),
        (
            "wildboar.embed._feature",
            ["src/wildboar/embed/_feature.pyx"],
            [],
        ),
        (
            "wildboar.embed._shapelet_fast",
            ["src/wildboar/embed/_shapelet_fast.pyx"],
            [],
        ),
        (
            "wildboar.embed._rocket_fast",
            ["src/wildboar/embed/_rocket_fast.pyx"],
            [],
        ),
        (
            "wildboar.embed._cinterval",
            ["src/wildboar/embed/_cinterval.pyx"],
            [],
        ),
        (
            "wildboar.embed._embed_fast",
            ["src/wildboar/embed/_embed_fast.pyx"],
            [],
        ),
        (
            "wildboar.embed.catch22._catch22",
            ["src/wildboar/embed/catch22/_catch22.pyx"],
            [],
        ),
        (
            "wildboar.utils._fft._pocketfft",
            [
                "src/wildboar/utils/_fft/_pocketfft.pyx",
                "src/wildboar/utils/_fft/src/pocketfft.c",
            ],
            ["src/wildboar/utils/_fft/src/"],
        ),
        (
            "wildboar.utils._stats",
            ["src/wildboar/utils/_stats.pyx"],
            [],
        ),
    ]

    ext_modules = [
        Extension(
            name,
            sources=sources,
            include_dirs=include_dirs + extra_include,
            **build_args
        )
        for name, sources, extra_include in extensions
    ]
    setup(ext_modules=cythonize(ext_modules))
