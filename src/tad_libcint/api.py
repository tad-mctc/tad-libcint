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

The shared objects are expected to be located in the same directory as this
module.
"""
from pathlib import Path

from tad_libcint.lazyloader import LazySharedLibraryLoader

__all__ = ["CINT", "CGTO"]


relpath = Path(__file__).parent.resolve()
CINT = LazySharedLibraryLoader("cint", relpath)
CGTO = LazySharedLibraryLoader("cgto", relpath)
# CPBC = LazySharedLibraryLoader("cpbc", relpath)  # currently not available
# CSYMM = LazySharedLibraryLoader("symm", relpath)  # currently not available
# CVHF = LazySharedLibraryLoader("CVHF", relpath)  # currently not available
