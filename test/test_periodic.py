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
Test the integrals of periodic structures: lattice sums at the Γ point and
at k-points, their gradients, and batches.

The reference is a ghost supercell: the rows of the central cell against the
atoms of all images within libcint's own threshold, evaluated by the
molecular driver and folded with the Bloch phases. PySCF provides an
independent reference for overlaps.
"""

from __future__ import annotations

import math
import tracemalloc
from typing import Any

import numpy as np
import pytest
import torch
from tad_mctc.io.structure import Structure, pack_structures
from tad_mctc.kpoints import KPoints, make_kpoints, pack_kpoints, to_cartesian
from tad_mctc.neighbor import build_periodic_shifts
from torch.autograd import gradcheck, gradgradcheck
from torch.func import jacfwd, jacrev, vmap

from tad_libcint import Basis, int1e, library, overlap
from tad_libcint.integrals import _pair_cutoff
from tad_libcint.names import IntegralName

from .data import (
    DTYPE,
    KCART,
    KFRAC,
    LATTICE,
    OH_CELL,
    WATER_CELL,
    ShellSpec,
    element_shells,
    make_shells,
    pyscf_ao_index,
    pyscf_shells,
)

dd: dict[str, Any] = {"dtype": DTYPE}


def _with_periodic(structure: Structure, periodic: list[bool]) -> Structure:
    """The structure periodic along some axes, with orthogonal vacuum axes"""
    assert structure.lattice is not None
    lattice = structure.lattice.clone()
    for axis, p in enumerate(periodic):
        if not p:
            lattice[axis] = 0.0
            lattice[:, axis] = 0.0
            lattice[axis, axis] = 5.0  # short: images within the cutoff
    return structure.replace(lattice=lattice, periodic=torch.tensor(periodic))


def _reference(
    name: IntegralName,
    basis: Basis,
    structure: Structure,
    kfrac: torch.Tensor | None = None,
    coeffs: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Lattice sums from a ghost supercell: the central cell's rows against all
    images within exp(-60) for the most diffuse primitives (with the basis'
    coefficients or `coeffs`)
    """
    if coeffs is None:
        coeffs = basis.coeffs
    lattice, periodic = structure.lattice, structure.periodic
    assert lattice is not None and periodic is not None
    numbers, positions = structure.numbers, structure.positions
    amin = float(basis.alphas.min())
    span = float(torch.cdist(positions, positions).max())
    cutoff = math.sqrt(2.0 * 60.0 / amin) + span
    shifts = torch.as_tensor(
        build_periodic_shifts(lattice, periodic, cutoff).shifts
    )
    shifts = shifts[(shifts[:, ~periodic] == 0).all(-1)]
    nimg = len(shifts)

    trans = shifts.to(DTYPE) @ lattice
    ext_numbers = numbers.repeat(nimg + 1)
    ext_positions = torch.cat(
        [positions, (positions[None] + trans[:, None]).reshape(-1, 3)]
    )
    layout = basis.layout(ext_numbers)
    nsh = layout.nsh // (nimg + 1)
    charges = ext_numbers.to(DTYPE)
    out = library.int2c(
        name.symbol(basis.spherical),
        name.unpacked.comp_shape,
        False,
        shls_slice=(0, nsh, nsh, layout.nsh),
        components=name.components,
        atm=layout.atm(charges),
        bas=layout.bas,
        env=layout.env(charges, ext_positions, basis.alphas, coeffs),
        ao_loc=layout.ao_loc,
    )
    nao = out.shape[-2]
    blocks = torch.from_numpy(out).reshape(*out.shape[:-1], nimg, nao)
    if kfrac is None:
        return blocks.sum(-2)
    phase = torch.exp(2j * math.pi * (kfrac @ shifts.T.to(DTYPE)))
    return torch.einsum("...inj,kn->k...ij", blocks.to(phase.dtype), phase)


def _name(name: str, packed: bool) -> IntegralName:
    iname = IntegralName.parse(name)
    return IntegralName(iname.op, packed=True) if packed else iname


def _as_real(out: torch.Tensor) -> torch.Tensor:
    """Real and imaginary part of complex integrals (k-points)"""
    return torch.view_as_real(out) if out.is_complex() else out


def _mesh_kpoints() -> list[KPoints]:
    """k-point meshes of :data:`WATER_CELL` and :data:`OH_CELL`"""
    assert WATER_CELL.lattice is not None and OH_CELL.lattice is not None
    return [
        make_kpoints(WATER_CELL.lattice, (2, 1, 1)),
        make_kpoints(OH_CELL.lattice, (1, 2, 2)),
    ]


# values


@pytest.mark.parametrize(
    "name, packed",
    [
        ("ovlp", False),
        ("j", False),
        ("jj", False),
        ("jj", True),
        ("jjj", False),
    ],
)
@pytest.mark.parametrize("kpoints", [False, True], ids=["gamma", "k"])
def test_reference(
    name: str, packed: bool, kpoints: bool, basis: Basis
) -> None:
    """Lattice sums equal the ghost supercell, at Γ and at k-points."""
    kfrac = KFRAC if kpoints else None
    kcart = KCART if kpoints else None
    out = int1e(
        name, basis, WATER_CELL, packed=packed, kpoints=kcart, precision=1e-14
    )
    ref = _reference(_name(name, packed), basis, WATER_CELL, kfrac)
    assert out.shape == ref.shape
    assert out.dtype == (torch.complex128 if kpoints else DTYPE)
    assert torch.allclose(out, ref, atol=1e-12)


@pytest.mark.parametrize(
    "periodic",
    [[True, True, False], [False, True, False], [False, False, False]],
)
def test_low_dimensional(periodic: list[bool], basis: Basis) -> None:
    """Images only along the periodic axes, however short the others."""
    cell = _with_periodic(WATER_CELL, periodic)
    assert cell.lattice is not None
    kfrac = KFRAC * torch.tensor(periodic, **dd)
    for name in ("ovlp", "jj"):
        iname = _name(name, packed=False)
        out = int1e(name, basis, cell, packed=False, precision=1e-14)
        assert torch.allclose(out, _reference(iname, basis, cell), atol=1e-12)
        out = int1e(
            name,
            basis,
            cell,
            packed=False,
            kpoints=to_cartesian(kfrac, cell.lattice),
            precision=1e-14,
        )
        ref = _reference(iname, basis, cell, kfrac)
        assert torch.allclose(out, ref, atol=1e-12)


def test_no_periodic_axes(basis: Basis) -> None:
    """A cell without periodic axes is the molecule."""
    cell = WATER_CELL.replace(periodic=torch.tensor([False] * 3))
    molecule = Structure(numbers=cell.numbers, positions=cell.positions)
    for name in ("ovlp", "j", "jj"):
        assert torch.allclose(
            int1e(name, basis, cell), int1e(name, basis, molecule), atol=1e-14
        )


@pytest.mark.parametrize("dimension", [3, 2, 1])
@pytest.mark.parametrize("spherical", [True, False], ids=["sph", "cart"])
def test_pyscf(dimension: int, spherical: bool) -> None:
    """Overlaps at Γ and at k-points equal PySCF's (images along the
    periodic axes only, `low_dim_ft_type="inf_vacuum"`)."""
    pbcgto = pytest.importorskip("pyscf.pbc.gto")
    periodic = [axis < dimension for axis in range(3)]
    cell = _with_periodic(WATER_CELL, periodic)
    assert cell.lattice is not None
    mol = pbcgto.M(
        atom=[
            (int(z), p.tolist()) for z, p in zip(cell.numbers, cell.positions)
        ],
        a=cell.lattice.numpy(),
        unit="Bohr",
        basis={
            "O": [
                [0, [12.0, 0.2], [2.5, 0.5], [0.6, 0.4]],
                [1, [3.0, 0.6], [0.7, 0.5]],
            ],
            "H": [[0, [3.4, 0.3], [0.6, 0.7]], [1, [0.8, 1.0]]],
        },
        cart=not spherical,
        dimension=dimension,
        low_dim_ft_type=None if dimension == 3 else "inf_vacuum",
        precision=1e-16,
        verbose=0,
    )

    shells = pyscf_shells(mol, cell.numbers.tolist())
    basis = Basis(shells=make_shells(shells), spherical=spherical)
    layout = basis.layout(cell.numbers)
    index = pyscf_ao_index(layout) if spherical else torch.arange(layout.nao)

    kfrac = KFRAC * torch.tensor(periodic, **dd)
    kcart = to_cartesian(kfrac, cell.lattice)
    ref = torch.from_numpy(
        np.asarray(mol.pbc_intor("int1e_ovlp", kpts=kcart.numpy()))
    )
    ref = ref[..., index, :][..., index]
    out = int1e("ovlp", basis, cell, kpoints=kcart, precision=1e-14)
    assert torch.allclose(out, ref, atol=1e-12)

    ref_gamma = torch.from_numpy(mol.pbc_intor("int1e_ovlp"))[index][:, index]
    out_gamma = int1e("ovlp", basis, cell, precision=1e-14)
    assert torch.allclose(out_gamma, ref_gamma, atol=1e-12)


def test_symmetry(basis: Basis) -> None:
    """The overlap is symmetric at Γ and hermitian at every k-point."""
    s = overlap(basis, WATER_CELL)
    assert torch.allclose(s, s.mT, atol=1e-14)
    sk = int1e("ovlp", basis, WATER_CELL, kpoints=KCART)
    assert torch.allclose(sk, sk.mT.conj(), atol=1e-14)
    j = int1e("j", basis, WATER_CELL)
    assert not torch.allclose(j, j.mT, atol=1e-3)


def test_wrapped_atom(basis: Basis) -> None:
    """
    Moving an atom by a lattice vector L changes no lattice sum at Γ, and at
    k-points multiplies its rows and columns by the Bloch phases,
    A'(k) = exp(i k (L_mu - L_nu)) A(k).
    """
    move = LATTICE[0] - 2 * LATTICE[2]
    positions = WATER_CELL.positions.clone()
    positions[1] += move
    moved = WATER_CELL.replace(positions=positions)

    kcart = KCART
    ao_atom = basis.layout(WATER_CELL.numbers).ao_to_atom
    phase = torch.exp(1j * (kcart @ move))[:, None] ** (ao_atom == 1)[None, :]
    for name in ("ovlp", "j", "jj"):
        a = int1e(name, basis, WATER_CELL)
        assert torch.allclose(a, int1e(name, basis, moved), atol=1e-13)

        a = int1e(name, basis, WATER_CELL, kpoints=kcart)
        b = int1e(name, basis, moved, kpoints=kcart)
        expected = a * (phase[:, :, None] * phase[:, None, :].conj()).reshape(
            len(kcart), *(1,) * (a.ndim - 3), *a.shape[-2:]
        )
        assert torch.allclose(b, expected, atol=1e-13)


def test_padding_kpoints(basis: Basis) -> None:
    """Padding k-points (masked) give zeros, the others their values."""
    packed = pack_kpoints([KPoints(points=KCART[:1]), KPoints(points=KCART)])
    assert packed.weights is not None and packed.mask is not None
    single = KPoints(
        points=packed.points.select(0, 0),
        weights=packed.weights.select(0, 0),
        mask=packed.mask.select(0, 0),
    )
    out = int1e("j", basis, WATER_CELL, kpoints=single)
    ref = int1e("j", basis, WATER_CELL, kpoints=KCART[:1])
    assert torch.equal(out[1], torch.zeros_like(out[1]))
    assert torch.allclose(out[:1], ref, atol=1e-14)


# precision


@pytest.mark.parametrize("precision", [1e-6, 1e-8, 1e-10])
def test_precision(precision: float) -> None:
    """The error of each element stays below the precision."""
    diffuse: dict[int, list[ShellSpec]] = {
        1: [
            (0, [3.4, 0.6, 0.08], [0.3, 0.7, 0.2]),
            (1, [0.8, 0.15], [1.0, 0.3]),
        ],
    }
    basis = Basis(shells=make_shells(diffuse)).normalized()
    cell = Structure(
        numbers=torch.tensor([1, 1]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], **dd),
        lattice=torch.tensor(
            [[9.0, 0.0, 0.0], [1.0, 9.5, 0.0], [0.0, 0.5, 10.0]], **dd
        ),
    )
    for name in ("ovlp", "j", "jj"):
        out = int1e(name, basis, cell, packed=False, precision=precision)
        ref = _reference(_name(name, packed=False), basis, cell)
        assert (out - ref).abs().max() <= precision
    # the cutoff does neglect images: a tighter precision changes the values
    loose = int1e("jj", basis, cell, precision=precision)
    tight = int1e("jj", basis, cell, precision=1e-14)
    assert (loose - tight).abs().max() > 0


def test_diffuse_cutoff_memory() -> None:
    """The cutoff scan of very diffuse primitives stays small in memory."""
    alphas = np.geomspace(1e-6, 1e-3, 6)
    coeffs = np.ones(6)
    tracemalloc.start()
    try:
        cut = _pair_cutoff(
            (alphas, coeffs, 1), (alphas, coeffs, 1), 2, 1e-11, (1000.0, 3)
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert 1e3 < cut < 1e5
    assert peak < 2**25


# zero coefficients


@pytest.mark.parametrize("zero", ["primitive", "shell", "all"])
def test_zero_coeffs(zero: str) -> None:
    """
    Values and coefficient gradients with zero coefficients equal the
    supercell's: a zero primitive or shell still screens the images of its
    derivatives. The integral is quadratic in the coefficients, so central
    differences are exact.
    """
    spec: dict[int, list[ShellSpec]] = {
        1: [(0, [1.2, 0.05], [0.7, 0.3]), (1, [0.8], [1.0])],
    }
    basis = Basis(shells=make_shells(spec)).normalized()
    cell = Structure(
        numbers=torch.tensor([1, 1]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], **dd),
        lattice=torch.tensor(
            [[9.0, 0.0, 0.0], [1.0, 9.5, 0.0], [0.0, 0.5, 10.0]], **dd
        ),
    )
    zeros = {"primitive": [1], "shell": [2], "all": [0, 1, 2]}[zero]
    coeffs = basis.coeffs.detach().clone()
    coeffs[zeros] = 0.0
    iname = _name("jj", packed=False)
    weights = torch.linspace(-1.0, 1.0, _reference(iname, basis, cell).numel())

    def reference(c: torch.Tensor) -> torch.Tensor:
        ref = _reference(iname, basis, cell, coeffs=c)
        return (weights * ref.flatten()).sum()

    x = coeffs.clone().requires_grad_(True)
    out = int1e("jj", basis, cell, packed=False, coeffs=x)
    ref = _reference(iname, basis, cell, coeffs=coeffs)
    assert (out - ref).abs().max() <= 1e-10
    (grad,) = torch.autograd.grad((weights * out.flatten()).sum(), x)

    h = 0.1
    step = torch.eye(len(coeffs), **dd)
    fd = torch.stack(
        [
            (reference(coeffs + h * e) - reference(coeffs - h * e)) / (2 * h)
            for e in step
        ]
    )
    assert torch.allclose(grad, fd, atol=1e-8)
    if zero != "all":
        assert grad[zeros].abs().max() > 1e-3


# padding


def test_all_padding(basis: Basis) -> None:
    """A cell of padding atoms has an empty integral, as a molecule does,
    and gives zeros in a batch."""
    positions = torch.zeros(2, 3, **dd, requires_grad=True)
    cell = Structure(
        numbers=torch.tensor([0, 0]), positions=positions, lattice=LATTICE
    )
    molecule = Structure(numbers=cell.numbers, positions=positions)
    out = int1e("jj", basis, cell)
    assert out.shape == int1e("jj", basis, molecule).shape
    (grad,) = torch.autograd.grad(out.sum(), positions, allow_unused=True)
    assert grad is None or torch.equal(grad, torch.zeros_like(grad))

    empty = Structure(
        numbers=torch.tensor([0]),
        positions=torch.zeros(1, 3, **dd),
        lattice=LATTICE,
    )
    batch = pack_structures([WATER_CELL, empty])
    out = int1e("jj", basis, batch)
    assert torch.allclose(out[0], int1e("jj", basis, WATER_CELL), atol=1e-14)
    assert torch.equal(out[1], torch.zeros_like(out[1]))


# gradients


def _cell_func(
    name: str, basis: Basis, structure: Structure, kpoints: bool, wrt: str
) -> Any:
    """The integral (real and imaginary part) as a function of one input"""
    assert structure.lattice is not None
    params = {
        "positions": structure.positions.clone(),
        "lattice": structure.lattice.clone(),
        "alphas": basis.alphas.detach().clone(),
        "coeffs": basis.coeffs.detach().clone(),
    }

    def func(x: torch.Tensor) -> torch.Tensor:
        p = {**params, wrt: x}
        cell = structure.replace(positions=p["positions"], lattice=p["lattice"])
        # fractional k-points held fixed, as the integral's derivatives are
        kcart = to_cartesian(KFRAC, p["lattice"]) if kpoints else None
        out = int1e(
            name,
            basis,
            cell,
            alphas=p["alphas"],
            coeffs=p["coeffs"],
            kpoints=kcart,
        )
        return _as_real(out)

    return func, params[wrt]


@pytest.mark.parametrize("name", ["ovlp", "j", "jj"])
@pytest.mark.parametrize("wrt", ["positions", "lattice", "alphas", "coeffs"])
@pytest.mark.parametrize("kpoints", [False, True], ids=["gamma", "k"])
def test_grad(name: str, wrt: str, kpoints: bool) -> None:
    """Gradients w.r.t. every input equal finite differences."""
    basis = Basis(shells=element_shells())
    func, x = _cell_func(name, basis, OH_CELL, kpoints, wrt)
    assert gradcheck(func, (x.requires_grad_(True),), atol=1e-7, fast_mode=True)


@pytest.mark.parametrize("name", ["ovlp", "jj"])
@pytest.mark.parametrize("wrt", ["positions", "lattice"])
def test_gradgrad(name: str, wrt: str) -> None:
    """Second derivatives w.r.t. positions and lattice."""
    basis = Basis(shells=element_shells())
    func, x = _cell_func(name, basis, OH_CELL, False, wrt)
    assert gradgradcheck(
        func, (x.requires_grad_(True),), atol=1e-6, fast_mode=True
    )


def test_no_periodic_axes_lattice_derivatives(basis: Basis) -> None:
    """Without periodic axes, nothing depends on the lattice."""
    cell = WATER_CELL.replace(periodic=torch.tensor([False] * 3))
    assert cell.lattice is not None

    def func(lattice: torch.Tensor) -> torch.Tensor:
        return int1e("jj", basis, cell.replace(lattice=lattice))

    zeros = torch.zeros(*func(cell.lattice).shape, 3, 3, **dd)
    assert torch.equal(jacrev(func)(cell.lattice), zeros)
    assert torch.equal(jacfwd(func)(cell.lattice), zeros)


def test_vacuum_gradient(basis: Basis) -> None:
    """A vacuum lattice vector has no gradient."""
    cell = _with_periodic(WATER_CELL, [True, False, True])
    assert cell.lattice is not None
    lattice = cell.lattice.clone().requires_grad_(True)
    out = int1e("jj", basis, cell.replace(lattice=lattice))
    (grad,) = torch.autograd.grad(out.pow(2).sum(), lattice)
    assert torch.equal(grad[1], torch.zeros(3, **dd))
    assert grad[0].abs().max() > 1e-6 and grad[2].abs().max() > 1e-6


@pytest.mark.parametrize("kpoints", [False, True], ids=["gamma", "k"])
def test_jacobians(kpoints: bool, basis: Basis) -> None:
    """jacrev and jacfwd agree, also for the Hessian of a scalar."""
    kcart = KCART if kpoints else None

    def func(positions: torch.Tensor, lattice: torch.Tensor) -> torch.Tensor:
        cell = WATER_CELL.replace(positions=positions, lattice=lattice)
        return _as_real(int1e("j", basis, cell, kpoints=kcart))

    args = (WATER_CELL.positions, LATTICE)
    for argnums in (0, 1):
        rev = jacrev(func, argnums=argnums)(*args)
        fwd = jacfwd(func, argnums=argnums)(*args)
        assert torch.allclose(rev, fwd, atol=1e-12)
        assert rev.abs().max() > 1e-3

    def energy(lattice: torch.Tensor) -> torch.Tensor:
        return func(WATER_CELL.positions, lattice).pow(2).sum()

    hessian = jacrev(torch.func.grad(energy))(LATTICE)
    assert torch.allclose(
        hessian, jacfwd(torch.func.grad(energy))(LATTICE), atol=1e-10
    )
    assert torch.allclose(
        hessian, hessian.reshape(9, 9).mT.reshape(3, 3, 3, 3), atol=1e-10
    )


# batches


def test_batch(basis: Basis) -> None:
    """A padded batch of different cells and k-point sets equals each one."""
    batch = pack_structures([WATER_CELL, OH_CELL])
    kpoints = _mesh_kpoints()
    out = int1e("j", basis, batch, kpoints=pack_kpoints(kpoints))
    for b, (cell, kpts) in enumerate(zip([WATER_CELL, OH_CELL], kpoints)):
        ref = int1e("j", basis, cell, kpoints=kpts)
        nk, _, nao = ref.shape[0], ref.shape[1], ref.shape[-1]
        assert torch.allclose(out[b, :nk, ..., :nao, :nao], ref, atol=1e-14)
        assert torch.equal(out[b, nk:], torch.zeros_like(out[b, nk:]))
        assert torch.equal(
            out[b, ..., nao:, :], torch.zeros_like(out[b, ..., nao:, :])
        )

    gamma = overlap(basis, batch)
    ref = overlap(basis, OH_CELL)
    nao = ref.shape[-1]
    assert torch.allclose(gamma[1, :nao, :nao], ref, atol=1e-14)


def test_vmap_jacobian(basis: Basis) -> None:
    """vmap over cells of the lattice Jacobian equals each cell's."""
    assert WATER_CELL.lattice is not None and OH_CELL.lattice is not None
    batch = pack_structures([WATER_CELL, OH_CELL])
    assert batch.lattice is not None

    def func(structure: Structure, lattice: torch.Tensor) -> torch.Tensor:
        return int1e("jj", basis, structure.replace(lattice=lattice))

    jac = vmap(jacrev(func, argnums=1))(batch, batch.lattice)
    for b, cell in enumerate([WATER_CELL, OH_CELL]):
        assert cell.lattice is not None
        ref = jacrev(func, argnums=1)(cell, cell.lattice)
        nao = ref.shape[-3]
        assert torch.allclose(jac[b, ..., :nao, :nao, :, :], ref, atol=1e-13)


# errors


def test_errors(basis: Basis) -> None:
    """Origin- and nucleus-dependent operators, misplaced k-points and
    precisions are refused."""
    for name in ("r0", "r0r0", "nuc", "ipnuc"):
        with pytest.raises(ValueError, match="no periodic integral"):
            int1e(name, basis, WATER_CELL)
    molecule = Structure(
        numbers=WATER_CELL.numbers, positions=WATER_CELL.positions
    )
    with pytest.raises(ValueError, match="periodic structure"):
        int1e("ovlp", basis, molecule, kpoints=torch.zeros(1, 3, **dd))
    with pytest.raises(ValueError, match="positive precision"):
        int1e("ovlp", basis, WATER_CELL, precision=0.0)
    with pytest.raises(ValueError, match="shape"):
        int1e("ovlp", basis, WATER_CELL, kpoints=torch.zeros(3, **dd))
    with pytest.raises(ValueError, match="one k-point set per structure"):
        int1e("ovlp", basis, WATER_CELL, kpoints=torch.zeros(2, 1, 3, **dd))

    cell = _with_periodic(WATER_CELL, [True, True, False])
    assert cell.lattice is not None
    with pytest.raises(ValueError, match="non-periodic axis"):
        int1e("ovlp", basis, cell, kpoints=to_cartesian(KFRAC, cell.lattice))


# more transforms


@pytest.mark.parametrize("wrt", ["alphas", "coeffs"])
@pytest.mark.parametrize("kpoints", [False, True], ids=["gamma", "k"])
def test_jacobians_parameters(wrt: str, kpoints: bool) -> None:
    """jacrev and jacfwd w.r.t. the basis parameters agree for a cell."""
    basis = Basis(shells=element_shells())
    func, x = _cell_func("jj", basis, OH_CELL, kpoints, wrt)
    rev = jacrev(func)(x)
    fwd = jacfwd(func)(x)
    assert torch.allclose(rev, fwd, atol=1e-12)
    assert rev.abs().max() > 1e-3


@pytest.mark.parametrize("kpoints", [False, True], ids=["gamma", "k"])
def test_gradgrad_mixed(kpoints: bool) -> None:
    """Second derivatives w.r.t. positions and lattice together."""
    basis = Basis(shells=element_shells())

    def func(positions: torch.Tensor, lattice: torch.Tensor) -> torch.Tensor:
        cell = OH_CELL.replace(positions=positions, lattice=lattice)
        kcart = to_cartesian(KFRAC, lattice) if kpoints else None
        return _as_real(int1e("j", basis, cell, kpoints=kcart))

    assert OH_CELL.lattice is not None
    args = (
        OH_CELL.positions.clone().requires_grad_(True),
        OH_CELL.lattice.clone().requires_grad_(True),
    )
    assert gradgradcheck(func, args, atol=1e-6, fast_mode=True)


def test_vmap_structures(basis: Basis) -> None:
    """`vmap` over a batched cell and k-point set equals the batch."""
    batch = pack_structures([WATER_CELL, OH_CELL])
    kpoints = pack_kpoints(_mesh_kpoints())

    def func(structure: Structure, kpts: KPoints) -> torch.Tensor:
        return int1e("j", basis, structure, kpoints=kpts)

    out = vmap(func)(batch, kpoints)
    assert torch.allclose(out, func(batch, kpoints), atol=1e-14)


def test_batch_mixed_periodicity(basis: Basis) -> None:
    """A batch of a 3D cell, a 2D slab and a cell without periodic axes."""
    slab = _with_periodic(WATER_CELL, [True, True, False])
    box = OH_CELL.replace(periodic=torch.tensor([False] * 3))
    cells = [WATER_CELL, slab, box]
    batch = pack_structures(cells)

    kpts = []
    for cell in cells:
        assert cell.lattice is not None and cell.periodic is not None
        kfrac = KFRAC * cell.periodic.to(DTYPE)
        kpts.append(KPoints(points=to_cartesian(kfrac, cell.lattice)))

    for kpoints in (None, pack_kpoints(kpts)):
        out = int1e("jj", basis, batch, kpoints=kpoints)
        for b, cell in enumerate(cells):
            ref = int1e(
                "jj",
                basis,
                cell,
                kpoints=None if kpoints is None else kpts[b],
            )
            nao = ref.shape[-1]
            assert torch.allclose(out[b, ..., :nao, :nao], ref, atol=1e-14)

    molecule = Structure(numbers=box.numbers, positions=box.positions)
    nao = int1e("jj", basis, molecule).shape[-1]
    assert torch.allclose(
        int1e("jj", basis, batch)[2, ..., :nao, :nao],
        int1e("jj", basis, molecule),
        atol=1e-14,
    )


@pytest.mark.large
def test_precision_ionic() -> None:
    """At the default precision, a rocksalt cell with a diffuse cation stays
    within it (the benchmark's LiF, here with the library's bound)."""
    shells: dict[int, list[ShellSpec]] = {
        3: [
            (0, [5.0, 0.6, 0.03], [0.3, 0.5, 0.3]),
            (1, [0.4, 0.025], [0.7, 0.4]),
        ],
        9: [
            (0, [20.0, 3.0, 0.8], [0.2, 0.5, 0.4]),
            (1, [4.0, 0.9, 0.25], [0.4, 0.5, 0.3]),
        ],
    }
    basis = Basis(shells=make_shells(shells)).normalized()
    a = 7.61
    cell = Structure(
        numbers=torch.tensor([3, 9]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [a / 2, a / 2, a / 2]], **dd),
        lattice=0.5 * a * torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]], **dd),
    )
    for name in ("ovlp", "j", "jj"):
        out = int1e(name, basis, cell, packed=False)
        ref = _reference(_name(name, packed=False), basis, cell)
        assert (out - ref).abs().max() <= 1e-10
