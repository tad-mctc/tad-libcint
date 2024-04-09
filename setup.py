# This file is part of tad-libcint, modified from diffqc/dqc.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Original file licensed under the Apache License, Version 2.0 by diffqc/dqc.
# Modifications made by Grimme Group.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Setup for C Extension.
"""

from pathlib import Path
from subprocess import call

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

module_name: str = "tad_libcint"
ext_name: str = "tad_libcint.pyscflibs"

file_dir: Path = Path(__file__).resolve().parent

print("")
print("")
print("TOP LEVEL")
print(file_dir)
call(["echo", " "])
call(["echo", file_dir])
call(["ls", "-la", file_dir])
print("")
print("")
print("")

# Arguments for CMake
CMAKE_ARGS = [
    "-DCMAKE_BUILD_TYPE=Release",
    "-DDISABLE_DFT=ON",
    "-DBUILD_LIBCINT=ON",
    "-DWITH_F12=OFF",
    "-DPYPZPX=OFF",
    # '-DWITH_RANGE_COULOMB=OFF', # no variable in CMakeLists.txt
]

# Number of cores for CMake's build step
CORES = 8


class CMakeExtension(Extension):
    """
    Extension module for CMake.
    """

    def __init__(self, name: str, sourcedir: str = "") -> None:
        """
        Initialize a new CMakeExtension.

        Parameters
        ----------
        name : str
            The name of the extension.
        sourcedir : str, optional
            The source directory of the extension.
        """
        super().__init__(name, sources=[])
        self.sourcedir = Path(sourcedir).resolve()


class CMakeBuildExt(build_ext):
    """
    Build extension module for CMake.
    """

    def run(self) -> None:
        """
        Run the extension building process.
        """
        extension = self.extensions[0]
        assert extension.name == ext_name
        self.build_extension(self.extensions[0])

    def build_extension(self, ext: Extension) -> None:
        """
        Build an extension.

        Parameters
        ----------
        ext : Extension
            The extension to build.
        """
        self.construct_extension(ext)

    def construct_extension(self, ext: Extension) -> None:
        """
        Construct an extension.

        Parameters
        ----------
        ext : Extension
            The extension to construct.
        """
        # libraries from PySCF
        lib_dir = file_dir / "src" / module_name / "libs"
        print("")
        print("")
        call(["echo", " "])
        call(["echo", "construct_extension"])
        call(["pwd"])
        print("")
        call(["ls", "-la"])
        call(["ls", "-la", file_dir])
        call(["ls", "-la", file_dir / "src"])
        call(["ls", "-la", file_dir / "src" / module_name])
        print("")
        print("")
        print("")

        build_dir = Path(self.build_temp)
        self.announce(
            f"Compiling libraries from PySCF from {lib_dir} to {build_dir}", level=3
        )
        self.build_cmake(ext, lib_dir, build_dir)

    def build_cmake(self, ext: Extension, lib_dir: Path, build_dir: Path) -> None:
        """
        Build an extension with CMake.

        Parameters
        ----------
        ext : Extension
            The extension to build.
        lib_dir : Path
            The library directory of the extension.
        build_dir : Path
            The build directory for CMake.
        """
        extdir = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        self.announce("Configuring cmake", level=3)
        cmake_args = ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + str(extdir)] + CMAKE_ARGS
        call(["ls", "-la", lib_dir])
        cmd = ["cmake", f"-S{lib_dir}", f"-B{build_dir}"] + cmake_args
        self.spawn(cmd)

        self.announce("Building binaries", level=3)
        cmd = ["cmake", "--build", str(build_dir), f"-j{CORES}"]
        self.spawn(cmd)


setup(
    ext_modules=[CMakeExtension(ext_name)],
    cmdclass={"build_ext": CMakeBuildExt},
)
