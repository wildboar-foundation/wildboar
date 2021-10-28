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

    def include_dirs(*args):
        return [np_include_dirs(), *args]

    build_args = BUILD_ARGS.get(BUILD_TYPE)
    if build_args is None:
        raise RuntimeError("%s is not a valid build type" % BUILD_TYPE)

    extensions = {
        "wildboar._utils": {
            "sources": ["src/wildboar/_utils.pyx"],
            "include_dirs": include_dirs(),
        },
        "wildboar.distance._distance": {
            "sources": ["src/wildboar/distance/_distance.pyx"],
            "include_dirs": include_dirs(),
        },
        "wildboar.distance._euclidean_distance": {
            "sources": ["src/wildboar/distance/_euclidean_distance.pyx"],
            "include_dirs": include_dirs(),
        },
        "wildboar.distance._dtw_distance": {
            "sources": ["src/wildboar/distance/_dtw_distance.pyx"],
            "include_dirs": include_dirs(),
        },
        "wildboar.tree._tree_builder": {
            "sources": ["src/wildboar/tree/_tree_builder.pyx"],
            "include_dirs": include_dirs(),
        },
        "wildboar._data": {
            "sources": ["src/wildboar/_data.pyx"],
            "include_dirs": include_dirs(),
        },
        "wildboar.embed._feature": {
            "sources": ["src/wildboar/embed/_feature.pyx"],
            "include_dirs": include_dirs(),
        },
        "wildboar.embed._shapelet_fast": {
            "sources": ["src/wildboar/embed/_shapelet_fast.pyx"],
            "include_dirs": include_dirs(),
        },
        "wildboar.embed._rocket_fast": {
            "sources": ["src/wildboar/embed/_rocket_fast.pyx"],
            "include_dirs": include_dirs(),
        },
        "wildboar.embed._cinterval": {
            "sources": ["src/wildboar/embed/_cinterval.pyx"],
            "include_dirs": include_dirs(),
        },
        "wildboar.embed._embed_fast": {
            "sources": ["src/wildboar/embed/_embed_fast.pyx"],
            "include_dirs": include_dirs(),
        },
        "wildboar.embed.catch22._catch22": {
            "sources": [
                "src/wildboar/embed/catch22/_catch22.pyx",
                "src/wildboar/embed/catch22/src/catch22.c",
            ],
            "include_dirs": include_dirs("src/wildboar/embed/catch22/src/"),
        },
        "wildboar.utils._fft._pocketfft": {
            "sources": [
                "src/wildboar/utils/_fft/_pocketfft.pyx",
                "src/wildboar/utils/_fft/src/pocketfft.c",
            ],
            "include_dirs": include_dirs("src/wildboar/utils/_fft/src/"),
        },
        "wildboar.utils._stats": {
            "sources": [
                "src/wildboar/utils/_stats.pyx",
                "src/wildboar/utils/src/stats.c",
            ],
            "include_dirs": include_dirs("src/wildboar/utils/src/"),
        },
    }
    ext_modules = [
        Extension(name, **options, **build_args) for name, options in extensions.items()
    ]

    setup(ext_modules=cythonize(ext_modules))
