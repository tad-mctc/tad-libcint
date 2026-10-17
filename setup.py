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

import os
import shlex
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

LIB_DIR = Path(__file__).resolve().parent / "src" / "tad_libcint" / "libs"

# Arguments for CMake, extended by the `CMAKE_ARGS` environment variable
# (e.g. `CMAKE_ARGS="-DBUILD_MARCH_NATIVE=ON" pip install .`)
CMAKE_ARGS = [
    "-DCMAKE_BUILD_TYPE=Release",
    *shlex.split(os.environ.get("CMAKE_ARGS", "")),
]


class CMakeBuildExt(build_ext):
    """Build the shared libraries in `LIB_DIR` with CMake"""

    def build_extension(self, ext: Extension) -> None:
        extdir = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        build_dir = Path(self.build_temp)
        self.spawn(
            [
                "cmake",
                f"-S{LIB_DIR}",
                f"-B{build_dir}",
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
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


setup(
    # the libraries are no Python extension; the name only sets their directory
    ext_modules=[Extension("tad_libcint._libraries", sources=[])],
    cmdclass={"build_ext": CMakeBuildExt},
)
