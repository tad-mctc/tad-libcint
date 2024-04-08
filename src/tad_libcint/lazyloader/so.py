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

from tad_libcint.typing import Any, Callable

__all__ = ["LazySharedLibraryLoader"]


class LazySharedLibraryLoader:
    """
    Lazy loader for shared objects. The ojects are expected to be callable.
    """

    def __init__(self, loader: Callable) -> None:
        self._loader = loader
        self._lib = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._lib is None:
            self._lib = self._loader()
        return self._lib(*args, **kwargs)
