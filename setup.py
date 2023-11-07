# -*- coding: utf-8 -*-

import os

from setuptools import setup
from setuptools.extension import Extension

BUILD_ARGS = {
    "default": {
        "posix": {
            "extra_compile_args": ["-O2"],
            "extra_link_args": [],
            "libraries": [],
        },
        "nt": {
            "extra_compile_args": ["/O2"],
            "extra_link_args": [],
            "libraries": [],
        },
    },
    "debug": {
        "posix": {
            "extra_compile_args": ["-O0", "-g"],
            "extra_link_args": [],
            "libraries": [],
        },
        "nt": {
            "extra_compile_args": ["/Od"],
            "extra_link_args": [],
            "libraries": [],
        },
    },
    "optimize": {
        "posix": {
            "extra_compile_args": [
                "-O3",
                "-march=native",
                "-ffast-math",
            ],
            "extra_link_args": [],
            "libraries": [],
        },
        "nt": {
            "extra_compile_args": [
                "/Ox",
                "/fp:fast",
            ],
            "extra_link_args": [],
            "libraries": [],
        },
    },
}


def make_extension(name, extension, defaults):
    import numpy

    include_dirs = extension.get("include_dirs", [])
    include_dirs.append(numpy.get_include())

    libraries = extension.get("libraries", [])
    libraries.extend(defaults["libraries"])

    extra_compile_args = extension.get("extra_compile_args", [])
    extra_compile_args.extend(defaults["extra_compile_args"])

    extra_link_args = extension.get("extra_link_args", [])
    extra_link_args.extend(defaults["extra_compile_args"])

    return Extension(
        name,
        sources=extension["sources"],
        include_dirs=include_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )


if __name__ == "__main__":
    from Cython.Build import cythonize

    build_type = os.getenv("WILDBOAR_BUILD", "default")
    build_nthreads = int(os.getenv("WILDBOAR_BUILD_NTHREADS", "1"))
    build_args = BUILD_ARGS.get(build_type, {}).get(os.name, None)

    if build_args is None:
        raise RuntimeError(f"{build_type} is not a valid build type")

    external_extra_compile_args = os.getenv("WILDBOAR_EXTRA_COMPILE_ARGS")
    if external_extra_compile_args is not None:
        import re

        build_args["extra_compile_args"].extend(
            re.split(r"\s+", external_extra_compile_args)
        )

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
        make_extension(name, options, build_args)
        for name, options in extensions.items()
    ]

    setup(
        ext_modules=cythonize(
            ext_modules,
            nthreads=build_nthreads,
            annotate=build_type == "debug",
        )
    )
