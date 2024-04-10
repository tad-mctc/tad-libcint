# This file is part of tad-libcint.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
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
LazyLoader: Shared Objects
==========================

Lazing loading of the *libcint* shared objects.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys
from pathlib import Path

from tad_libcint.typing import Any, Callable

__all__ = ["LazySharedLibraryLoader"]


class LazySharedLibraryLoader:
    """
    Lazy loader for shared objects.
    """

    name: str
    """The name of the shared object."""

    path: str | Path
    """The path to the shared object."""

    lib: ctypes.CDLL | None
    """The shared object library."""

    def __init__(self, name: str, relpath: str | Path) -> None:
        # Determine the library extension based on the operating system
        ext = "dylib" if sys.platform == "darwin" else "so"

        self.path = Path(relpath) / f"lib{name.casefold()}.{ext}"
        self.name = name
        self.lib = None

    def __getattr__(self, func_name: str) -> Callable[..., Any]:
        if self.lib is None:
            try:
                self.lib = ctypes.cdll.LoadLibrary(str(self.path))
            except OSError as exc:
                path2 = ctypes.util.find_library(self.name)
                if path2 is None:
                    raise exc
                self.lib = ctypes.cdll.LoadLibrary(path2)

        return getattr(self.lib, func_name)
