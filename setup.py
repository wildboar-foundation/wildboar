# -*- coding: utf-8 -*-

import sys
import os
import ast

from setuptools import setup
from setuptools.extension import Extension


def build_ext(*args, **kwargs):
    import Cython.Build
    return Cython.Build.build_ext(*args, **kwargs)


libname = "wildboar"
build_type = "optimized"

SHORTDESC = "wildboar is the fundamental package for time series classification with Python"

DESC = """
It provides:

 * Shapelet tree classification and regression
 * Random shapelet forest classification and regression
 * Fast dynamic time warning searching
 * Fast euclidean distance searching

The package is provided under the GPLv3 license.
"""

datadirs = ("test",)
dataexts = (".py", ".pyx", ".pxd", ".c", ".cpp", ".h", ".sh", ".lyx", ".tex",
            ".txt", ".pdf")

standard_docs = ["README", "LICENSE", "TODO", "CHANGELOG", "AUTHORS"]
standard_doc_exts = [".md", ".rst", ".txt", "", ".org"]

if sys.version_info < (3, 4):
    sys.exit('Sorry, Python < 3.4 is not supported')

extra_compile_args_math_optimized = [
    '-march=native',
    '-O2',
    '-msse',
    '-msse2',
    '-mfma',
    '-mfpmath=sse',
]
extra_compile_args_math_debug = [
    '-march=native',
    '-O0',
    '-g',
]

extra_link_args_math_optimized = []
extra_link_args_math_debug = []

extra_compile_args_nonmath_optimized = ['-O2']
extra_compile_args_nonmath_debug = ['-O0', '-g']
extra_link_args_nonmath_optimized = []
extra_link_args_nonmath_debug = []

openmp_compile_args = ['-fopenmp']
openmp_link_args = ['-fopenmp']


# Lazy loading
class np_include_dirs(str):
    def __str__(self):
        import numpy as np
        return np.get_include()


my_include_dirs = [np_include_dirs()]

if build_type == 'optimized':
    my_extra_compile_args_math = extra_compile_args_math_optimized
    my_extra_compile_args_nonmath = extra_compile_args_nonmath_optimized
    my_extra_link_args_math = extra_link_args_math_optimized
    my_extra_link_args_nonmath = extra_link_args_nonmath_optimized
    my_debug = False
    print("build configuration selected: optimized")
elif build_type == 'debug':
    my_extra_compile_args_math = extra_compile_args_math_debug
    my_extra_compile_args_nonmath = extra_compile_args_nonmath_debug
    my_extra_link_args_math = extra_link_args_math_debug
    my_extra_link_args_nonmath = extra_link_args_nonmath_debug
    my_debug = True
    print("build configuration selected: debug")
else:
    raise ValueError(
        "Unknown build configuration '%s'; valid: 'optimized', 'debug'" %
        (build_type))


def declare_cython_extension(extName,
                             use_math=False,
                             use_openmp=False,
                             include_dirs=None,
                             extra_lib=None):
    extPath = extName.replace(".", os.path.sep) + ".pyx"
    if use_math:
        compile_args = list(my_extra_compile_args_math)  # copy
        link_args = list(my_extra_link_args_math)
        libraries = ["m"]
    else:
        compile_args = list(my_extra_compile_args_nonmath)
        link_args = list(my_extra_link_args_nonmath)
        libraries = None

    if use_openmp:
        compile_args.insert(0, openmp_compile_args)
        link_args.insert(0, openmp_link_args)

    if extra_lib is not None:
        if libraries is None:
            libraries = []
        for lib in extra_lib:
            libraries.append(lib)

    return Extension(
        extName, [extPath],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=include_dirs,
        libraries=libraries)


datafiles = []


def getext(filename):
    os.path.splitext(filename)[1]


for datadir in datadirs:
    datafiles.extend(
        [(root,
          [os.path.join(root, f) for f in files if getext(f) in dataexts])
         for root, dirs, files in os.walk(datadir)])

detected_docs = []
for docname in standard_docs:
    for ext in standard_doc_exts:
        filename = "".join((docname, ext))
        if os.path.isfile(filename):
            detected_docs.append(filename)
datafiles.append(('.', detected_docs))

init_py_path = os.path.join(libname, '__init__.py')
version = '0.0.unknown'
try:
    with open(init_py_path) as f:
        for line in f:
            if line.startswith('__version__'):
                version = ast.parse(line).body[0].value.s
                break
        else:
            print(
                "WARNING: Version information not found"
                " in '%s', using placeholder '%s'" % (init_py_path, version),
                file=sys.stderr)
except FileNotFoundError:
    print(
        "WARNING: Could not find file '%s',"
        "using placeholder version information '%s'" % (init_py_path, version),
        file=sys.stderr)

ext_module_utils = declare_cython_extension(
    "wildboar._utils",
    use_math=True,
    use_openmp=False,
    include_dirs=my_include_dirs)

ext_module_distance = declare_cython_extension(
    "wildboar._distance",
    use_math=True,
    use_openmp=False,
    include_dirs=my_include_dirs)

ext_module_euclidean_distance = declare_cython_extension(
    "wildboar._euclidean_distance",
    use_math=True,
    use_openmp=False,
    include_dirs=my_include_dirs)

ext_module_dtw_distance = declare_cython_extension(
    "wildboar._dtw_distance",
    use_math=True,
    use_openmp=False,
    include_dirs=my_include_dirs)

# ext_module_mass_distance = declare_cython_extension(
#     "wildboar._mass_distance",
#     use_math=True,
#     use_openmp=False,
#     include_dirs=my_include_dirs,
#     extra_lib=["fftw3"])

ext_module_impurity = declare_cython_extension(
    "wildboar._impurity",
    use_math=True,
    use_openmp=False,
    include_dirs=my_include_dirs)

ext_module_tree_builder = declare_cython_extension(
    "wildboar._tree_builder",
    use_math=True,
    use_openmp=False,
    include_dirs=my_include_dirs)

ext_module_distance_api = declare_cython_extension(
    "wildboar.distance",
    use_math=False,
    use_openmp=False,
    include_dirs=my_include_dirs)

cython_ext_modules = [
    ext_module_utils,
    ext_module_distance,
    ext_module_euclidean_distance,
    ext_module_dtw_distance,
    #    ext_module_mass_distance,
    ext_module_distance_api,
    ext_module_impurity,
    ext_module_tree_builder,
]

setup(
    name="wildboar",
    version=version,
    author="Isak Samsten",
    author_email="isak@samsten.se",
    url="https://github.com/isakkarlsson/wildboar",
    description=SHORTDESC,
    long_description=DESC,
    license="GPLv3",
    #    platforms=["Linux"],
    classifiers=[
        "Development Status :: 4 - Beta",
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
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
    ],
    setup_requires=[
        'cython>=0.28',
        'numpy>=1.14.2',
        'setuptools>=18.0',
    ],
    install_requires=["scikit-learn"],
    python_requires=">=3.4.0",
    provides=["wildboar"],
    keywords=["machine learning", "time series distance"],
    ext_modules=cython_ext_modules,
    packages=["wildboar"],
    package_data={
        'wildboar': ['*.pxd', '*.pyx', '*.c'],
    },
    zip_safe=False,
    #    cmdclass={'build_ext': build_ext},

    # Custom data files not inside a Python package
    data_files=datafiles,
)
