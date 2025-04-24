from setuptools import setup
from Cython.Build import cythonize

setup(
    name="core_search",
    ext_modules=cythonize("core_search.pyx",
                          language_level=3,
                          compiler_directives={"boundscheck": False,
                                               "wraparound": False}),
    zip_safe=False,
)
