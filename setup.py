from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(
    name="mpeb_c",
    ext_modules=cythonize("mpeb_c.pyx", annotate=True , include_path=[numpy.get_include()]),
)
