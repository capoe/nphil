import os
import sys
import platform
import glob
from setuptools import setup, find_packages, Extension

class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user
    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

def get_scripts():
    return [ "bin/philter" ]

def find_env_include_dir():
    pydir = os.path.dirname(sys.executable)
    return os.path.join(pydir, "..", "include")

def make_cxx_extensions():
    srcs = sorted(glob.glob('nphil/cxx/*.cpp'))
    incdirs = [
        "./nphil/cxx",
        "./external",
        find_env_include_dir(),
        get_pybind_include(),
        get_pybind_include(user=True)]
    cpp_extra_link_args = [
        "-lmkl_rt"
    ]
    cpp_extra_compile_args = [
        "-std=c++11", 
        "-O3", 
        "-Wno-reorder", 
        "-Wno-sign-compare", 
        "-Wno-return-type"]
    c_extra_compile_args = [
        "-std=c99", 
        "-O3"]
    return [
        Extension(
            'nphil._nphil',
            srcs,
            include_dirs=incdirs,
            libraries=[],
            language='c++',
            extra_compile_args=cpp_extra_compile_args + ["-fvisibility=hidden"],
            extra_link_args=cpp_extra_link_args,
        )
    ]

def get_description():
    return "Non-parametric non-linear feature network filtering and feature generation for sparse data modelling"

if __name__ == "__main__":
    setup(
        name="nphil",
        version="0.1.4",
        author="capoe",
        author_email="carl.poelking@floatlab.io",
        url="https://github.com/capoe/nphil",
        description="NPHIL",
        long_description=get_description(),
        scripts=get_scripts(),
        packages=find_packages(),
        setup_requires=['pybind11>=2.4', 'mkl', 'mkl-include'],
        install_requires=['pybind11>=2.4', "numpy", "scipy", "scikit-learn", "mkl", "mkl-include"],
        include_package_data=True,
        ext_modules=make_cxx_extensions(),
        license="GNU General Public License (Version 2)",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Physics",
            "Topic :: Scientific/Engineering :: Chemistry",
            "Topic :: Scientific/Engineering :: Artificial Intelligence"
        ],
        keywords="3d convolutional descriptors chemical machine learning",
        python_requires=">=3.0",
    )

