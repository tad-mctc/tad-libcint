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
Interface: Symmetry
===================

Collection of symmetry classes.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tad_libcint.interface.symmetry import s1 as s1
    from tad_libcint.interface.symmetry import s4 as s4
else:
    import tad_libcint.lazyloader as _lazy

    __getattr__, __dir__, __all__ = _lazy.attach_module(
        __name__,
        ["s1", "s4"],
    )

    del _lazy

del TYPE_CHECKING
