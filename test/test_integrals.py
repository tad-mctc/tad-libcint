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
Test the values and gradients of the integrals.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from typing import Any

import pytest
import torch
from tad_mctc.data.structures import get_structure
from tad_mctc.io.structure import Structure
from torch.autograd import gradcheck, gradgradcheck

from tad_libcint import Basis, Shell, int1e, overlap
from tad_libcint.basis import Layout

from .data import (
    DTYPE,
    NUMBERS,
    OH,
    POSITIONS,
    WATER,
    ShellSpec,
    element_shells,
    gto_norm,
    make_shells,
)


def _primitive(angmom: int, alpha: float) -> Shell:
    """Normalized primitive shell"""
    return Shell(
        angmom,
        torch.tensor([alpha], dtype=DTYPE),
        torch.tensor([gto_norm(angmom, alpha)], dtype=DTYPE),
    )


# values


@pytest.mark.parametrize("spherical", [True, False])
def test_normalized_primitives(spherical: bool) -> None:
    """Normalized s, p and (spherical) d primitives have unit norm."""
    angmoms = [0, 1, 2] if spherical else [0, 1]
    basis = Basis(
        {1: [_primitive(l, 0.8) for l in angmoms]}, spherical=spherical
    )

    atom = Structure(
        numbers=torch.tensor([1]), positions=torch.zeros((1, 3), dtype=DTYPE)
    )
    s = overlap(basis, atom)
    assert torch.allclose(s.diagonal(), torch.ones(len(s), dtype=DTYPE))


def test_overlap_two_s_primitives() -> None:
    """Analytic overlap of two normalized s Gaussians."""
    a, b, dist = 0.7, 1.9, 1.3
    basis = Basis({1: [_primitive(0, a)], 2: [_primitive(0, b)]})
    structure = Structure(
        numbers=torch.tensor([1, 2]),
        positions=torch.tensor([[0, 0, 0], [0, 0, dist]], dtype=DTYPE),
    )

    ref = (2 * math.sqrt(a * b) / (a + b)) ** 1.5 * math.exp(
        -a * b / (a + b) * dist**2
    )
    s = overlap(basis, structure)
    assert s[0, 1].item() == pytest.approx(ref)


def test_ket_centred_multipoles(basis: Basis) -> None:
    """``j``, ``jj`` and ``jjj`` are powers of ``r - R_ket``."""
    s, r0, r0r0, r0r0r0, j, jj, jjj = (
        int1e(name, basis, WATER)
        for name in ("ovlp", "r0", "r0r0", "r0r0r0", "j", "jj", "jjj")
    )
    layout = basis.layout(NUMBERS)
    ket = POSITIONS[layout.ao_to_atom].T[:, None, :]  # (3, 1, nao)

    assert torch.allclose(j, r0 - ket * s)

    nao = layout.nao
    r0r0, jj = r0r0.reshape(3, 3, nao, nao), jj.reshape(3, 3, nao, nao)
    ref = (
        r0r0
        - ket[:, None] * r0[None]
        - ket[None] * r0[:, None]
        + ket[:, None] * ket[None] * s
    )
    assert torch.allclose(jj, ref)

    # (r - R)^3 = rrr - 3 R rr + 3 R R r - R R R, symmetrized over a, b, c
    r0r0r0 = r0r0r0.reshape(3, 3, 3, nao, nao)
    jjj = jjj.reshape(3, 3, 3, nao, nao)
    ra, rb, rc = ket[:, None, None], ket[None, :, None], ket[None, None, :]
    ref = (
        r0r0r0
        - ra * r0r0[None, :, :]
        - rb * r0r0[:, None, :]
        - rc * r0r0[:, :, None]
        + ra * rb * r0[None, None, :]
        + ra * rc * r0[None, :, None]
        + rb * rc * r0[:, None, None]
        - ra * rb * rc * s
    )
    assert torch.allclose(jjj, ref)


@pytest.mark.parametrize("name, rank", [("jj", 2), ("jjj", 3)])
def test_multipole_symmetric(name: str, rank: int, basis: Basis) -> None:
    """The components of ``jj`` and ``jjj`` are symmetric tensors."""
    out = int1e(name, basis, WATER)
    nao = out.shape[-1]
    out = out.reshape(*(3,) * rank, nao, nao)
    for perm in itertools.permutations(range(rank)):
        assert torch.allclose(out, out.permute(*perm, rank, rank + 1))


def test_ket_derivative_by_transpose() -> None:
    """
    ``r0ip`` is not in libcint and comes from transposing ``ipr0``. Check it
    against the derivative of ``r0`` w.r.t. the position of the ket atom.
    """
    basis = Basis(element_shells())
    positions = POSITIONS[:2]
    ao_to_atom = basis.layout(NUMBERS[:2]).ao_to_atom
    bra, ket = ao_to_atom == 0, ao_to_atom == 1

    def r0(ket_position: torch.Tensor) -> torch.Tensor:
        pos = torch.stack([positions[0], ket_position])
        return int1e("r0", basis, OH.replace(positions=pos))[:, bra][..., ket]

    # (3, nbra, nket, 3): derivative w.r.t. the ket centre is minus nabla
    jac = torch.autograd.functional.jacobian(r0, positions[1])
    r0ip = int1e("r0ip", basis, OH)[:, :, bra][..., ket]
    assert torch.allclose(-jac.movedim(-1, 1), r0ip)


def test_fractional_charge() -> None:
    """Nuclear attraction is linear in each (fractional) nuclear charge."""
    basis = Basis(element_shells())

    def nuc(charge: float) -> torch.Tensor:
        charges = torch.tensor([1.0, charge])
        return int1e("nuc", basis, OH, nuclear_charges=charges)

    assert torch.allclose(nuc(0.5), 0.5 * (nuc(0.0) + nuc(1.0)))
    assert not torch.allclose(nuc(0.5), nuc(1.0))

    # integer charges keep their sign
    assert torch.allclose(nuc(-1.0) - nuc(0.0), nuc(0.0) - nuc(1.0))


PYSCF_INTEGRALS = [
    ("ovlp", "int1e_ovlp"),
    ("nuc", "int1e_nuc"),
    ("r0", "int1e_r"),
    ("r0r0", "int1e_rr"),
    ("j", "int1e_r_origj"),
    ("jj", "int1e_rr_origj"),
    ("r0r0r0", "int1e_rrr"),
]


@pytest.mark.parametrize("name, intor", PYSCF_INTEGRALS)
@pytest.mark.parametrize("spherical", [True, False], ids=["sph", "cart"])
def test_pyscf(name: str, intor: str, spherical: bool) -> None:
    """Agreement with PySCF's independent libcint build."""
    shells = [[0, [3.4, 0.3], [0.6, 0.7]], [1, [0.8, 1.0]]]
    basis = {"O": shells, "H": shells}
    _compare_pyscf(name, intor, WATER, basis, spherical)


@pytest.mark.large
@pytest.mark.parametrize("name, intor", PYSCF_INTEGRALS)
@pytest.mark.parametrize("spherical", [True, False], ids=["sph", "cart"])
def test_pyscf_large(name: str, intor: str, spherical: bool) -> None:
    """Agreement with PySCF for MB16_43_01 in a basis with d functions."""
    structure = get_structure("mb16_43", "01")
    _compare_pyscf(name, intor, structure, "def2-svp", spherical)


def _compare_pyscf(
    name: str,
    intor: str,
    structure: Structure,
    pyscf_basis: Any,
    spherical: bool,
) -> None:
    """Compare an integral with PySCF, using PySCF's normalized coefficients"""
    gto = pytest.importorskip("pyscf.gto")
    mol = gto.M(
        atom=[
            (int(z), pos.tolist())
            for z, pos in zip(structure.numbers, structure.positions)
        ],
        unit="Bohr",
        basis=pyscf_basis,
        cart=not spherical,
        spin=None,
    )

    per_atom: list[list[ShellSpec]] = [[] for _ in range(mol.natm)]
    for ish in range(mol.nbas):
        # pylint: disable-next=protected-access
        coeffs = mol._libcint_ctr_coeff(ish)[:, 0].tolist()
        shell = (int(mol.bas_angular(ish)), mol.bas_exp(ish).tolist(), coeffs)
        per_atom[mol.bas_atom(ish)].append(shell)

    # PySCF places shells per atom, the basis per element
    elements: dict[int, list[ShellSpec]] = {}
    for z, shells in zip(structure.numbers.tolist(), per_atom):
        assert elements.setdefault(z, shells) == shells
    basis = Basis(make_shells(elements), spherical=spherical)

    ref = torch.from_numpy(mol.intor(intor))
    if spherical:
        # PySCF orders spherical p functions as (px, py, pz)
        index = _pyscf_ao_index(basis.layout(structure.numbers))
        ref = ref[..., index, :][..., index]
    out = int1e(name, basis, structure)
    assert torch.allclose(out, ref, atol=1e-11)


def _pyscf_ao_index(layout: Layout) -> torch.Tensor:
    """PySCF's AO index of each of our spherical AOs"""
    index = torch.arange(layout.nao)
    for ish in range(layout.nsh):
        if layout.shell_angmom[ish] == 1:
            first = int(layout.ao_loc[ish])
            index[first : first + 3] = first + torch.tensor([1, 2, 0])
    return index


def test_spherical_p_order() -> None:
    """Spherical p functions are ordered (py, pz, px), cartesian (x, y, z)."""
    s = _primitive(0, 1.0)
    p = _primitive(1, 1.0)
    structure = Structure(
        numbers=torch.tensor([1, 2]),
        positions=torch.tensor([[0, 0, 0], [1.0, 0, 0]], dtype=DTYPE),
    )

    # only the p function pointing towards the s function (x) overlaps
    for spherical, px in ((True, 2), (False, 0)):
        basis = Basis({1: [p], 2: [s]}, spherical=spherical)
        overlaps = overlap(basis, structure)[:3, 3].abs()
        assert overlaps.argmax().item() == px
        assert overlaps.count_nonzero().item() == 1


def test_batched_structure(basis: Basis, conformer_batch: Structure) -> None:
    """A batch of structures is evaluated one by one."""
    out = int1e("r0", basis, conformer_batch)

    nao = basis.layout(NUMBERS).nao
    assert out.shape == (3, 3, nao, nao)
    for i, pos in enumerate(conformer_batch.positions):
        ref = int1e("r0", basis, WATER.replace(positions=pos))
        assert torch.equal(out[i], ref)


def test_batched_nuclear_charges(
    basis: Basis, conformer_batch: Structure
) -> None:
    """Nuclear charges are shared by a batch or given per structure."""
    batch = conformer_batch
    charges = torch.tensor([7.5, 1.0, 1.0], dtype=DTYPE)

    shared = int1e("nuc", basis, batch, nuclear_charges=charges)
    each = int1e("nuc", basis, batch, nuclear_charges=charges.expand(3, -1))
    assert torch.equal(shared, each)
    assert not torch.allclose(shared, int1e("nuc", basis, batch))


# input validation


def test_invalid_inputs(basis: Basis) -> None:
    """Shapes and dtypes of the inputs are checked."""
    with pytest.raises(NotImplementedError, match="nuclear charges"):
        charges = NUMBERS.double().requires_grad_(True)
        int1e("nuc", basis, WATER, nuclear_charges=charges)
    with pytest.raises(ValueError, match="nuclear charges of shape"):
        int1e("nuc", basis, WATER, nuclear_charges=NUMBERS[:2])
    with pytest.raises(ValueError, match="alphas of shape"):
        overlap(basis, WATER, alphas=basis.alphas[:-1])
    with pytest.raises(ValueError, match="one dtype"):
        overlap(basis, WATER.type(torch.float32))
    with pytest.raises(ValueError, match="Cannot parse"):
        int1e("kin", basis, WATER)
    with pytest.raises(ValueError, match="no shells for elements"):
        overlap(Basis({1: element_shells()[1]}), WATER)


@pytest.mark.parametrize("periodic", [True, False])
def test_periodic_unsupported(periodic: bool, basis: Basis) -> None:
    """Structures with a lattice are refused, even without periodic axes."""
    cell = WATER.replace(
        lattice=10 * torch.eye(3, dtype=DTYPE),
        periodic=torch.full((3,), periodic),
    )
    with pytest.raises(NotImplementedError, match="periodic"):
        overlap(basis, cell)


def test_missing_integral(basis: Basis) -> None:
    """Integrals that libcint lacks raise a clear error."""
    with pytest.raises(NotImplementedError, match="not available"):
        int1e("ipr0r0r0", basis, WATER)


# gradients


def _check(name: str, basis: Basis, wrt: str, second: bool = False) -> None:
    """(Second order) gradient check w.r.t. one of the parameters"""
    params = {
        "positions": POSITIONS.clone(),
        "alphas": basis.alphas.detach().clone(),
        "coeffs": basis.coeffs.detach().clone(),
    }

    def func(x: torch.Tensor) -> torch.Tensor:
        p = {**params, wrt: x}
        return int1e(
            name,
            basis,
            WATER.replace(positions=p["positions"]),
            alphas=p["alphas"],
            coeffs=p["coeffs"],
        )

    x = params[wrt].requires_grad_(True)
    check = gradgradcheck if second else gradcheck
    assert check(func, (x,), atol=1e-6, fast_mode=True)


OPERATORS = ["ovlp", "r0", "r0r0", "j", "jj"]


@pytest.mark.grad
@pytest.mark.parametrize("name", OPERATORS)
@pytest.mark.parametrize("wrt", ["positions", "alphas", "coeffs"])
def test_grad(name: str, wrt: str, basis: Basis) -> None:
    """First derivatives w.r.t. positions, exponents and coefficients."""
    _check(name, basis, wrt)


@pytest.mark.grad
@pytest.mark.parametrize("name", OPERATORS)
@pytest.mark.parametrize("wrt", ["positions", "alphas", "coeffs"])
def test_gradgrad(name: str, wrt: str) -> None:
    """Second derivatives w.r.t. positions, exponents and coefficients."""
    basis = Basis(element_shells())
    _check(name, basis, wrt, second=True)


@pytest.mark.grad
@pytest.mark.parametrize("name", ["nuc", "r0r0r0", "jjj"])
def test_grad_coeffs_forward_only(name: str, basis: Basis) -> None:
    """Coefficient gradients need no derivative integrals."""
    _check(name, basis, "coeffs")


@pytest.mark.grad
@pytest.mark.parametrize("name", ["ovlp", "j"])
def test_grad_zero_coefficient(name: str, basis: Basis) -> None:
    """A vanishing coefficient has a finite gradient."""
    coeffs = basis.coeffs.detach().clone()
    coeffs[0] = 0.0
    coeffs.requires_grad_(True)
    out = int1e(name, basis, WATER, coeffs=coeffs)
    (grad,) = torch.autograd.grad(out.pow(2).sum(), coeffs)
    assert grad.isfinite().all()
    assert grad[0] != 0

    def func(c: torch.Tensor) -> torch.Tensor:
        return int1e(name, basis, WATER, coeffs=c)

    assert gradgradcheck(func, (coeffs,), atol=1e-6, fast_mode=True)


@pytest.mark.parametrize("name", ["r0r0r0", "jjj"])
@pytest.mark.parametrize("wrt", ["positions", "alphas"])
def test_grad_forward_only_unavailable(
    name: str, wrt: str, basis: Basis
) -> None:
    """Integrals without derivatives in libcint refuse these gradients."""
    params = {
        "positions": POSITIONS.clone(),
        "alphas": basis.alphas.detach().clone(),
    }
    params[wrt].requires_grad_(True)
    positions, alphas = params["positions"], params["alphas"]
    out = int1e(name, basis, WATER.replace(positions=positions), alphas=alphas)
    with pytest.raises(NotImplementedError, match="not available"):
        out.sum().backward()


# derivatives of forces and Hessians (third order and parameter training)


def _small_basis() -> tuple[Basis, Structure]:
    """Two atoms with s and p shells, to keep the numerical checks cheap"""

    shells = make_shells(
        {
            8: [(0, [2.5, 0.6], [0.5, 0.4]), (1, [0.9], [1.0])],
            1: [(0, [1.2], [0.7])],
        }
    )
    return Basis(shells), OH


def _forces(
    name: str,
    basis: Basis,
    structure: Structure,
    order: int,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Derivative of order 1 (forces) or 2 (Hessian) of a fixed contraction of
    the integral w.r.t. the positions.
    """
    weight = torch.randn(
        int1e(name, basis, structure).shape,
        dtype=DTYPE,
        generator=torch.Generator().manual_seed(0),
    )

    def energy(
        positions: torch.Tensor, alphas: torch.Tensor, coeffs: torch.Tensor
    ) -> torch.Tensor:
        out = int1e(
            name,
            basis,
            structure.replace(positions=positions),
            alphas=alphas,
            coeffs=coeffs,
        )
        return (weight * out).sum()

    def derivative(
        positions: torch.Tensor, alphas: torch.Tensor, coeffs: torch.Tensor
    ) -> torch.Tensor:
        def func(pos: torch.Tensor) -> torch.Tensor:
            return energy(pos, alphas, coeffs)

        if order == 1:
            return torch.autograd.functional.jacobian(
                func, positions, create_graph=True
            )
        return torch.autograd.functional.hessian(
            func, positions, create_graph=True
        )

    return derivative


@pytest.mark.grad
@pytest.mark.parametrize("name", OPERATORS)
def test_third_order_positions(name: str) -> None:
    """Second derivatives of the forces, i.e. third order in the positions."""
    basis, structure = _small_basis()
    forces = _forces(name, basis, structure, order=1)
    positions = structure.positions.clone()

    def func(positions: torch.Tensor) -> torch.Tensor:
        return forces(positions, basis.alphas.detach(), basis.coeffs.detach())

    positions.requires_grad_(True)
    assert gradgradcheck(func, (positions,), atol=1e-6)


@pytest.mark.grad
@pytest.mark.parametrize("name", OPERATORS)
@pytest.mark.parametrize("order", [1, 2], ids=["forces", "hessian"])
def test_parameter_training(name: str, order: int) -> None:
    """Gradients of forces and Hessians w.r.t. exponents and coefficients."""
    basis, structure = _small_basis()
    derivative = _forces(name, basis, structure, order)
    positions = structure.positions

    alphas = basis.alphas.detach().clone().requires_grad_(True)
    coeffs = basis.coeffs.detach().clone().requires_grad_(True)
    assert gradcheck(
        lambda a, c: derivative(positions, a, c), (alphas, coeffs), atol=1e-6
    )


def test_grad_nuc_positions_unsupported(basis: Basis) -> None:
    """The nuclear attraction's position gradient is refused, not wrong."""
    positions = POSITIONS.clone().requires_grad_(True)
    out = int1e("nuc", basis, WATER.replace(positions=positions))
    with pytest.raises(NotImplementedError, match="nuclear positions"):
        out.sum().backward()
