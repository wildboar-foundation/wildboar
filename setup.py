from distutils.core import setup
from Cython.Build import cythonize
import Cython
import numpy

Cython.Compiler.Options.annotate = True

setup(
    ext_modules=cythonize([
        "pypf/_utils.pyx", "pypf/_sliding_distance.pyx", "pypf/_impurity.pyx",
        "pypf/_tree_builder.pyx"
    ]),
    include_dirs=[numpy.get_include()],
    extra_compile_args=["-O3"])
