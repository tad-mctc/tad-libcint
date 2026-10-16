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
PyTorch-based Libcint Interface
===============================

Differentiable one-electron integrals from the *libcint* high-performance
integral library. Gradients with respect to atomic positions, exponents and
contraction coefficients are computed by libcint's derivative integrals.

A :class:`Basis` holds the shells of every element, the integral functions
take a :class:`~tad_mctc.io.structure.Structure` (atomic numbers and
positions):

>>> import torch
>>> from tad_mctc.io.structure import Structure
>>> from tad_libcint import Basis, Shell, int1e
>>> dd = {"dtype": torch.double}
>>> h = [Shell(0, torch.tensor([1.0], **dd), torch.tensor([2.52647511], **dd))]
>>> basis = Basis({1: h})
>>> numbers = torch.tensor([1, 1])
>>> positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], **dd)
>>> h2 = Structure(numbers=numbers, positions=positions)
>>> int1e("r0", basis, h2).shape
torch.Size([3, 2, 2])
"""

from ._version import __version__
from .basis import Basis, Shell
from .integrals import int1e, overlap
from .library import CGTO, CINT

__all__ = ["Basis", "Shell", "int1e", "overlap", "CINT", "CGTO", "__version__"]
