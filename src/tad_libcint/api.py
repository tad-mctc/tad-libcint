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
API
===

Interface to the library objects from *libcint*. The shared libraries are
loaded lazily when the first attribute is accessed.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys

from .lazyloader import LazySharedLibraryLoader
from .typing import Any, Callable

__all__ = ["CINT", "CGTO"]


_libs: dict[str, Any] = {}


def _library_loader(name: str, relpath: str) -> Callable:
    curpath = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(os.path.join(curpath, relpath))

    # load the library and cache the handler
    def fcn():
        if name not in _libs:
            try:
                _libs[name] = ctypes.cdll.LoadLibrary(path)
            except OSError as exc:
                path2 = ctypes.util.find_library(name)
                if path2 is None:
                    raise exc
                _libs[name] = ctypes.cdll.LoadLibrary(path2)
        return _libs[name]

    return fcn


# libraries
ext = "dylib" if sys.platform == "darwin" else "so"


# Wrap the loaders with the lazy loading class
CINT = LazySharedLibraryLoader(_library_loader("cint", f"libcint.{ext}"))
CGTO = LazySharedLibraryLoader(_library_loader("cgto", f"libcgto.{ext}"))
CPBC = LazySharedLibraryLoader(_library_loader("cpbc", f"libpbc.{ext}"))
CSYMM = LazySharedLibraryLoader(_library_loader("symm", f"libsymm.{ext}"))
CVHF = LazySharedLibraryLoader(_library_loader("CVHF", f"libcvhf.{ext}"))
