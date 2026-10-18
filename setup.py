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

import importlib.util
import os
import shlex
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:  # setuptools < 70.1
    from wheel.bdist_wheel import bdist_wheel

LIB_DIR = Path(__file__).resolve().parent / "src" / "tad_libcint" / "libs"

# Arguments for CMake, extended by the `CMAKE_ARGS` environment variable
# (e.g. `CMAKE_ARGS="-DBUILD_MARCH_NATIVE=ON" pip install .`)
CMAKE_ARGS = [
    "-DCMAKE_BUILD_TYPE=Release",
    *shlex.split(os.environ.get("CMAKE_ARGS", "")),
]

# On Windows, CMake defaults to Visual Studio, but libcint needs a C99
# compiler (complex numbers): MinGW-w64's GCC, unless chosen otherwise
if (
    sys.platform == "win32"
    and "CMAKE_GENERATOR" not in os.environ
    and not any(arg.startswith("-G") for arg in CMAKE_ARGS)
):
    CMAKE_ARGS.append("-GMinGW Makefiles")


def editable_openmp_args() -> list[str]:
    """
    On macOS, libcgto loads PyTorch's libomp from next to the installed
    package (see libs/CMakeLists.txt), where an editable install has none.
    It takes the one of the building Python instead, which is only visible
    without build isolation, and otherwise builds without OpenMP.
    """
    if sys.platform != "darwin":
        return []
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        return ["-DENABLE_OPENMP=OFF"]
    torch_dir = Path(spec.submodule_search_locations[0])
    return [f"-DTORCH_LIBRARY_DIR={torch_dir / 'lib'}"]


class CMakeBuildExt(build_ext):
    """Build the shared libraries in `LIB_DIR` with CMake"""

    def build_extension(self, ext: Extension) -> None:
        extdir = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        build_dir = Path(self.build_temp)
        editable = self.inplace or getattr(self, "editable_mode", False)
        self.spawn(
            [
                "cmake",
                f"-S{LIB_DIR}",
                f"-B{build_dir}",
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
                # DLLs are runtime artifacts
                f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={extdir}",
                *(editable_openmp_args() if editable else []),
                *CMAKE_ARGS,
            ]
        )
        jobs = str(os.cpu_count() or 1)
        self.spawn(["cmake", "--build", str(build_dir), "--parallel", jobs])

    def copy_extensions_to_source(self) -> None:
        # Editable installs (and `--inplace` builds) copy every extension next
        # to its sources. The extension is only a placeholder that does not
        # exist, so copy the libraries that CMake built in its place.
        build_py = self.get_finalized_command("build_py")
        for ext in self.extensions:
            package = ext.name.rpartition(".")[0]
            source_dir = Path(build_py.get_package_dir(package))
            build_dir = (
                Path(self.build_lib) / self.get_ext_filename(ext.name)
            ).parent
            for library in build_dir.glob("lib*_tad.*"):
                self.copy_file(str(library), str(source_dir / library.name))


class PlatformWheel(bdist_wheel):
    """
    One wheel per platform for all Python versions (``py3-none-<platform>``):
    the libraries are loaded with ctypes and do not use Python's C API.
    """

    def get_tag(self) -> tuple[str, str, str]:
        platform = super().get_tag()[2]
        return "py3", "none", platform


setup(
    # the libraries are no Python extension; the name only sets their directory
    ext_modules=[Extension("tad_libcint._libraries", sources=[])],
    cmdclass={"build_ext": CMakeBuildExt, "bdist_wheel": PlatformWheel},
)
