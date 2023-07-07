import os
import sys
import re
import subprocess as sp
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

# module's descriptor
module_name = "dxtblibs"
ext_name = "dxtblibs.pyscflibs"
github_url = "https://github.com/grimme-lab/dxtblibs/tree/main/"
raw_github_url = "https://raw.githubusercontent.com/grimme-lab/dxtblibs/main/"

file_dir = os.path.dirname(os.path.realpath(__file__))
absdir = lambda p: os.path.join(file_dir, p)

# get the long description from README
# open readme and convert all relative path to absolute path
with open("README.md", "r") as f:
    long_desc = f.read()

link_pattern = re.compile(r"\(([\w\-/]+)\)")
img_pattern  = re.compile(r"\(([\w\-/\.]+)\)")
link_repl = r"(%s\1)" % github_url
img_repl  = r"(%s\1)" % raw_github_url
long_desc = re.sub(link_pattern, link_repl, long_desc)
long_desc = re.sub(img_pattern, img_repl, long_desc)

############### versioning ###############
verfile = os.path.abspath(os.path.join(module_name, "_version.py"))
version = {"__file__": verfile}
with open(verfile, "r") as fp:
    exec(fp.read(), version)

# execute _version.py to create _version.txt
cmd = [sys.executable, verfile]
sp.run(cmd)

############## build extensions ##############

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuildExt(build_ext):
    def run(self):
        extension = self.extensions[0]
        assert extension.name == ext_name
        self.build_extension(self.extensions[0])

    def build_extension(self, ext):
        self.construct_extension(ext)

    def construct_extension(self, ext):
        # libraries from PySCF
        lib_dir = os.path.join(file_dir, "dxtblibs", "libs")
        build_dir = self.build_temp
        self.announce(f'Compiling libraries from PySCF from {lib_dir} to {build_dir}', level=3)
        self.build_cmake(ext, lib_dir, build_dir)

    def build_cmake(self, ext, lib_dir, build_dir):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        self.announce("Configuring cmake", level=3)
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DCMAKE_BUILD_TYPE=Release',
            '-DDISABLE_DFT=ON',
            '-DBUILD_LIBCINT=ON',
            '-DWITH_F12=OFF',
            #'-DWITH_RANGE_COULOMB=OFF', # no variable in CMakeLists.txt
        ]
        cmd = ['cmake', f'-S{lib_dir}', f'-B{build_dir}'] + cmake_args
        self.spawn(cmd)

        self.announce("Building binaries", level=3)
        cmd = ['cmake', '--build', build_dir, '-j']
        self.spawn(cmd)

vers = version["get_version"]()
setup(
    name=module_name,
    version=vers,
    description='Libraries for dxtb',
    url='https://github.com/grimme-lab/dxtblibs/',
    long_description=long_desc,
    long_description_content_type="text/markdown",
    author='marvinfriede',
    author_email='friede@thch.uni-bonn.de',
    license='Apache License 2.0',
    packages=find_packages(),
    package_data={module_name: ["_version.txt"]},
    python_requires=">=3.8",
    install_requires=[
        # "numpy>=1.8.2",
        # "scipy>=0.15",
    ],
    ext_modules=[CMakeExtension(ext_name, '')],
    cmdclass={'build_ext': CMakeBuildExt},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: Apache Software License",

        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="project library tight-binding quantum-chemistry",
    zip_safe=False
)
