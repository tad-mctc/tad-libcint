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
Test inputs on a CUDA device: libcint runs on the CPU, but the results and
gradients must match the CPU's and live on the inputs' device.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.io.structure import Structure, pack_structures
from tad_mctc.kpoints import to_cartesian
from torch import Tensor
from torch.func import jacrev

from tad_libcint import Basis, int1e, integrals

from .data import DIMER, METHANE, WATER, element_shells
from .test_periodic import KFRAC, WATER_CELL

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)

CUDA = torch.device("cuda")


def _basis(device: torch.device) -> Basis:
    """The test basis with its parameters on `device`"""
    return Basis(shells=element_shells()).to(device=device)


def _to(structure: Structure, device: torch.device) -> Structure:
    return Structure(
        numbers=structure.numbers.to(device),
        positions=structure.positions.to(device),
    )


def _gradients(
    name: str, structure: Structure, device: torch.device
) -> list[Tensor]:
    """The integral and its gradients w.r.t. positions and parameters"""
    basis = _basis(device)
    structure = _to(structure, device)
    positions = structure.positions.clone().requires_grad_(True)
    alphas = basis.alphas.clone().requires_grad_(True)
    coeffs = basis.coeffs.clone().requires_grad_(True)
    out = int1e(
        name,
        basis,
        structure.replace(positions=positions),
        alphas=alphas,
        coeffs=coeffs,
    )
    grads = torch.autograd.grad(out.pow(2).sum(), (positions, alphas, coeffs))
    return [out, *grads]


@pytest.mark.parametrize("name", ["ovlp", "j", "jj"])
def test_gradients(name: str) -> None:
    """Integrals and gradients on CUDA match the CPU, on the CUDA device."""
    expected = _gradients(name, WATER, torch.device("cpu"))
    for out, ref in zip(_gradients(name, WATER, CUDA), expected):
        assert out.device.type == "cuda"
        assert torch.allclose(out.cpu(), ref)


def test_nuc() -> None:
    """The nuclear attraction on CUDA matches the CPU."""
    expected = int1e("nuc", _basis(torch.device("cpu")), WATER)
    out = int1e("nuc", _basis(CUDA), _to(WATER, CUDA))
    assert out.device.type == "cuda"
    assert torch.allclose(out.cpu(), expected)


def test_screening(monkeypatch: pytest.MonkeyPatch) -> None:
    """Screened shell pairs (neighbour search) work for CUDA inputs."""
    expected = _gradients("j", DIMER, torch.device("cpu"))
    monkeypatch.setattr(integrals, "_SCREENING_MIN_PAIRS", 0)
    for out, ref in zip(_gradients("j", DIMER, CUDA), expected):
        assert out.device.type == "cuda"
        assert torch.allclose(out.cpu(), ref)


def test_batch() -> None:
    """A padded batch of different molecules on CUDA matches the CPU."""
    batch = pack_structures([WATER, METHANE])
    expected = int1e("j", _basis(torch.device("cpu")), batch)
    out = int1e("j", _basis(CUDA), _to(batch, CUDA))
    assert out.device.type == "cuda"
    assert torch.allclose(out.cpu(), expected)


def test_jacobian() -> None:
    """`jacrev` (vmap over cotangents) on CUDA matches the CPU."""

    def jacobian(device: torch.device) -> Tensor:
        basis, structure = _basis(device), _to(WATER, device)

        def func(positions: Tensor) -> Tensor:
            return int1e("j", basis, structure.replace(positions=positions))

        return jacrev(func)(structure.positions)

    out = jacobian(CUDA)
    assert out.device.type == "cuda"
    assert torch.allclose(out.cpu(), jacobian(torch.device("cpu")))


@pytest.mark.parametrize("kpoints", [False, True], ids=["gamma", "k"])
def test_periodic(kpoints: bool) -> None:
    """Lattice sums and their lattice gradient on CUDA match the CPU."""

    def lattice_sums(device: torch.device) -> list[Tensor]:
        cell = WATER_CELL.to(device=device)
        assert cell.lattice is not None
        lattice = cell.lattice.clone().requires_grad_(True)
        kcart = to_cartesian(KFRAC.to(device), lattice) if kpoints else None
        out = int1e(
            "jj", _basis(device), cell.replace(lattice=lattice), kpoints=kcart
        )
        (grad,) = torch.autograd.grad(out.abs().pow(2).sum(), lattice)
        return [out, grad]

    expected = lattice_sums(torch.device("cpu"))
    for out, ref in zip(lattice_sums(CUDA), expected):
        assert out.device.type == "cuda"
        assert torch.allclose(out.cpu(), ref)
