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
Test the basis set and the layout of molecules.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from tad_mctc.io.structure import Structure

from tad_libcint import Basis, Shell, overlap

from .data import DTYPE, NUMBERS, POSITIONS, WATER, element_shells, gto_norm


def test_dimensions() -> None:
    """O has s, p, d and H has s, p shells (spherical: 9 + 2 * 4 AOs)."""
    basis = Basis(shells=element_shells())
    sph = basis.layout(NUMBERS)
    cart = Basis(shells=element_shells(), spherical=False).layout(NUMBERS)

    assert basis.nparam == 5 + 3 + 6  # C, H, O
    assert (sph.nat, sph.nsh, sph.nprim) == (3, 7, 12)
    assert sph.nao == 1 + 3 + 5 + 2 * (1 + 3)
    assert cart.nao == 1 + 3 + 6 + 2 * (1 + 3)
    assert sph.ao_to_atom.tolist() == [0] * 9 + [1] * 4 + [2] * 4
    assert sph.ao_to_shell.tolist()[:10] == [0, 1, 1, 1, 2, 2, 2, 2, 2, 3]
    assert repr(basis) == "Basis(elements=[1, 6, 8], nsh=7, spherical=True)"
    assert repr(sph) == "Layout(nat=3, nsh=7, nao=17, spherical=True)"


def test_padding() -> None:
    """Atoms with atomic number 0 are padding without shells."""
    basis = Basis(shells=element_shells())
    layout = basis.layout(NUMBERS)
    padded = basis.layout(torch.tensor([8, 1, 1, 0, 0]))

    assert padded.nat == 5
    assert padded.nao == layout.nao
    assert torch.equal(padded.ao_to_atom, layout.ao_to_atom)


def test_layout_cache() -> None:
    """
    Layouts are cached per list of atomic numbers and shared by all bases
    with the same shell structure, e.g. copies and converted bases.
    """
    basis = Basis(shells=element_shells())
    water = basis.layout(NUMBERS)
    assert basis.layout(NUMBERS.clone()) is water
    assert Basis(shells=element_shells()).layout(NUMBERS) is water
    assert basis.to(dtype=torch.float32).layout(NUMBERS) is water
    assert basis.layout(torch.tensor([1, 1])) is not water

    # not shared with a different structure
    cart = Basis(shells=element_shells(), spherical=False)
    assert cart.layout(NUMBERS) is not water


def test_libcint_arrays() -> None:
    """The pointers in `atm` and `bas` address the right `env` entries."""
    basis = Basis(shells=element_shells())
    layout = basis.layout(NUMBERS)
    atm = layout.atm(NUMBERS)
    env = layout.env(NUMBERS, POSITIONS, basis.alphas, basis.coeffs)

    for iat in range(layout.nat):
        ptr = atm[iat, 1]
        assert env[ptr : ptr + 3].tolist() == POSITIONS[iat].tolist()

    first = 0
    for ish in range(layout.nsh):
        nprim = layout.bas[ish, 2]
        params = layout.prim_param[first : first + nprim]
        exps = env[layout.bas[ish, 5] : layout.bas[ish, 5] + nprim]
        coeffs = env[layout.bas[ish, 6] : layout.bas[ish, 6] + nprim]
        assert exps.tolist() == basis.alphas[params].tolist()
        assert coeffs.tolist() == basis.coeffs[params].tolist()
        first += nprim


def test_fractional_charges() -> None:
    """All charges are read from `env` by libcint's fractional nuclear model."""
    basis = Basis(shells=element_shells())
    layout = basis.layout(NUMBERS)
    charges = torch.tensor([8.0, 0.5, -1.0])
    atm = layout.atm(charges)
    assert atm[:, 2].tolist() == [3, 3, 3]

    env = layout.env(charges, POSITIONS, basis.alphas, basis.coeffs)
    assert env[atm[:, 4]].tolist() == [8.0, 0.5, -1.0]


def test_shared_element_parameters() -> None:
    """Atoms of one element share the element's shell tensors."""
    shells = element_shells()
    alpha_h = shells[1][0].alphas.requires_grad_(True)
    basis = Basis(shells=shells)

    overlap(basis, WATER).sum().backward()
    assert alpha_h.grad is not None
    assert alpha_h.grad.shape == (2,)


def test_uncontracted() -> None:
    """Contracting the uncontracted integral recovers the contracted one."""
    basis = Basis(shells=element_shells())
    layout = basis.layout(NUMBERS)
    ulayout = basis.uncontracted.layout(NUMBERS)

    assert ulayout.nsh == ulayout.nprim == layout.nprim
    assert torch.equal(basis.uncontracted.alphas, basis.alphas)

    s = overlap(basis, WATER)
    us = overlap(basis.uncontracted, WATER)
    index = layout.uao_to_ao
    contracted = torch.zeros_like(s).index_put_(
        (index[:, None], index[None, :]), us, accumulate=True
    )
    assert torch.allclose(contracted, s)


@pytest.mark.parametrize("spherical, lmax", [(True, 4), (False, 1)])
def test_normalized(spherical: bool, lmax: int) -> None:
    """Normalized shells have unit self-overlap (Cartesian ones up to p)."""
    contracted = element_shells()[8][0]
    shells = [contracted.replace(angmom=l) for l in range(lmax + 1)]
    basis = Basis(shells={1: shells}, spherical=spherical).normalized()
    atom = Structure(
        numbers=torch.tensor([1]), positions=torch.zeros((1, 3), dtype=DTYPE)
    )

    diagonal = overlap(basis, atom).diagonal()
    assert torch.allclose(diagonal, torch.ones_like(diagonal))


def test_normalized_primitive() -> None:
    """A normalized primitive has libcint's normalization; it is idempotent."""
    for l in range(5):
        shell = Shell(
            angmom=l,
            alphas=torch.tensor([0.8], dtype=DTYPE),
            coeffs=torch.tensor([1.0], dtype=DTYPE),
        ).normalized()
        assert shell.coeffs.item() == pytest.approx(gto_norm(l, 0.8))
        assert torch.allclose(shell.normalized().coeffs, shell.coeffs)


def test_normalized_is_overlap_normalization() -> None:
    """
    Normalizing the shells equals normalizing the overlap by its diagonal,
    for the values and the gradient w.r.t. exponents and coefficients.
    """
    shells = element_shells()
    alphas = shells[8][2].alphas.requires_grad_(True)  # the d shell of O
    coeffs = shells[8][2].coeffs.requires_grad_(True)
    basis = Basis(shells=shells)

    s = overlap(basis, WATER)
    norm = s.diagonal().rsqrt()
    expected = s * norm[:, None] * norm[None, :]
    out = overlap(basis.normalized(), WATER)
    assert torch.allclose(out, expected)

    grads = torch.autograd.grad(out.pow(2).sum(), (alphas, coeffs))
    expected_grads = torch.autograd.grad(
        expected.pow(2).sum(), (alphas, coeffs)
    )
    for grad, ref in zip(grads, expected_grads):
        assert torch.allclose(grad, ref)


def test_missing_element() -> None:
    """Elements without shells are reported."""
    basis = Basis(shells=element_shells())
    with pytest.raises(ValueError, match=r"no shells for elements \[7\]"):
        basis.layout(torch.tensor([8, 7]))
    with pytest.raises(ValueError, match=r"shape \(nat,\)"):
        basis.layout(NUMBERS[None])


def test_invalid_basis() -> None:
    """The basis checks its shells."""
    with pytest.raises(ValueError, match="at least one shell"):
        Basis(shells={1: []})
    with pytest.raises(ValueError, match="positive"):
        Basis(shells={0: element_shells()[1]})

    single = Shell(
        angmom=0, alphas=torch.tensor([1.0]), coeffs=torch.tensor([1.0])
    )
    double = single.to(dtype=DTYPE)
    with pytest.raises(TypeError, match="different dtypes"):
        Basis(shells={1: [single], 2: [double]})


@pytest.mark.parametrize(
    "angmom, alphas, coeffs, match",
    [
        (-1, [1.0], [1.0], "Negative angular momentum"),
        (0, [], [], "non-empty 1D"),
        (0, [[1.0]], [[1.0]], "non-empty 1D"),
        (0, [1.0, 2.0], [1.0], "differ"),
    ],
)
def test_invalid_shell(
    angmom: int, alphas: list[Any], coeffs: list[Any], match: str
) -> None:
    """Shells validate their angular momentum and primitive tensors."""
    with pytest.raises(ValueError, match=match):
        Shell(
            angmom=angmom,
            alphas=torch.tensor(alphas),
            coeffs=torch.tensor(coeffs),
        )
