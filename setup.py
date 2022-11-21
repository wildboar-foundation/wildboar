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
        "wildboar.distance.*": {
            "sources": ["src/wildboar/distance/*.pyx"],
        },
        "wildboar.tree.*": {
            "sources": ["src/wildboar/tree/*.pyx"],
        },
        "wildboar.transform.*": {
            "sources": ["src/wildboar/transform/*.pyx"],
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
        "wildboar.utils.*": {"sources": ["src/wildboar/utils/*.pyx"]},
    }
    ext_modules = [
        Extension(name, **_merge_build_options(name, options, build_args))
        for name, options in extensions.items()
    ]

    setup(ext_modules=cythonize(ext_modules))
