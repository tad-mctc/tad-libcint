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
Test the integrals under `torch.compile`.

libcint is called through ctypes inside custom autograd functions, which
Dynamo cannot trace: the integrals compile with graph breaks, with values and
gradients (`.backward()`) equal to eager mode. Compiling them as one graph
(`fullgraph=True`) and compiling `torch.func` transforms of them fail; these
tests record that, so that a change in either direction is noticed.

The backend is ``aot_eager``, which traces with Dynamo and AOTAutograd without
generating code (independent of a C++ toolchain).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from tad_mctc.io.structure import Structure
from tad_mctc.kpoints import to_cartesian

from tad_libcint import Basis, int1e

from .data import WATER, element_shells
from .test_periodic import KFRAC, WATER_CELL

BACKEND = "aot_eager"


def _energy(
    case: str,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], Structure]:
    """A scalar function of the positions of a molecule or a cell"""
    basis = Basis(shells=element_shells())
    structure = WATER if case == "molecule" else WATER_CELL
    kpoints = None
    if case == "cell-k":
        assert WATER_CELL.lattice is not None
        kpoints = to_cartesian(KFRAC, WATER_CELL.lattice)

    def energy(positions: torch.Tensor) -> torch.Tensor:
        cell = structure.replace(positions=positions)
        out = int1e("jj", basis, cell, kpoints=kpoints)
        return (out.abs() if out.is_complex() else out).pow(2).sum()

    return energy, structure


CASES = ["molecule", "cell", "cell-k"]


@pytest.mark.parametrize("case", CASES)
def test_value(case: str) -> None:
    """Compiled with graph breaks, the value equals eager mode."""
    torch._dynamo.reset()  # pylint: disable=protected-access
    energy, structure = _energy(case)
    compiled = torch.compile(energy, backend=BACKEND)
    assert torch.equal(
        compiled(structure.positions), energy(structure.positions)
    )


@pytest.mark.parametrize("case", CASES)
def test_backward(case: str) -> None:
    """Compiled with graph breaks, `.backward()` equals eager mode."""
    torch._dynamo.reset()  # pylint: disable=protected-access
    energy, structure = _energy(case)
    expected = torch.func.grad(energy)(structure.positions)

    positions = structure.positions.clone().requires_grad_(True)
    torch.compile(energy, backend=BACKEND)(positions).backward()
    assert positions.grad is not None
    assert torch.allclose(positions.grad, expected, atol=1e-14)


@pytest.mark.xfail(
    strict=True,
    reason="libcint is called through ctypes in custom autograd functions, "
    "which Dynamo cannot trace as one graph",
)
@pytest.mark.parametrize("case", CASES)
def test_fullgraph(case: str) -> None:
    """One graph (`fullgraph=True`) is not supported."""
    torch._dynamo.reset()  # pylint: disable=protected-access
    energy, structure = _energy(case)
    torch.compile(energy, backend=BACKEND, fullgraph=True)(structure.positions)


@pytest.mark.xfail(
    strict=True,
    reason="Dynamo fails on the custom autograd functions inside "
    "`torch.func` transforms",
)
@pytest.mark.parametrize("case", CASES)
def test_func_transform(case: str) -> None:
    """Compiling `torch.func.grad` of an integral is not supported."""
    torch._dynamo.reset()  # pylint: disable=protected-access
    energy, structure = _energy(case)
    torch.compile(torch.func.grad(energy), backend=BACKEND)(structure.positions)
