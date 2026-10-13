from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# -ffp-contract=off: the AH encoder must round as numpy does (no fused
# multiply-add), so its codes match ScaNN's byte for byte.
ext = Pybind11Extension(
    "qa_scann._core",
    ["src/qa_scann/_core.cpp"],
    cxx_std=17,
    extra_compile_args=["-O3", "-ffp-contract=off"],
)

setup(ext_modules=[ext], cmdclass={"build_ext": build_ext})
