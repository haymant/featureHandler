from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup
import numpy

BASE_DIR = Path(__file__).resolve().parent
TARGET_DIR = BASE_DIR / "_libs"
TARGET_DIR.mkdir(parents=True, exist_ok=True)
# ensure we build from parent directory so output path `featureHandler/_libs` is correct
import os
os.chdir(BASE_DIR.parent)

extensions = [
    Extension(
        "featureHandler._libs.rolling",
        [str(BASE_DIR / "_libs" / "rolling.pyx")],
        include_dirs=[numpy.get_include()],
        language="c++",
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    )
]

setup(
    name="featureHandler_ext",
    ext_modules=cythonize(extensions, language_level=3),
    # run with this script from anywhere; build output will land in {}
    options={"build_ext": {"build_lib": str(TARGET_DIR.parent)}},
)
