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
Test the composition with `torch.func` transforms, over conformers of one
molecule and over padded batches of different molecules.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import pytest
import torch
from tad_mctc.io.structure import Structure
from torch import Tensor
from torch.autograd.functional import hessian, jacobian
from torch.func import grad, jacrev, vmap

from tad_libcint import Basis, int1e, overlap

from .data import METHANE, WATER


def _loop(func: Callable[[Tensor], Tensor], batch: Tensor) -> Tensor:
    """Reference: apply `func` to each batch element in a Python loop"""
    return torch.stack([func(x) for x in batch])


def _at(name: str, basis: Basis) -> Callable[[Tensor], Tensor]:
    """The integral `name` of water as a function of the positions"""

    def func(positions: Tensor) -> Tensor:
        return int1e(name, basis, WATER.replace(positions=positions))

    return func


MOLECULES = (WATER, METHANE)
"""The unpadded molecules of the `mixed` batch"""


# conformers of one molecule


@pytest.mark.parametrize("name", ["ovlp", "r0", "jj"])
def test_vmap_positions(name: str, basis: Basis, conformers: Tensor) -> None:
    """Every conformer gets its own integrals (not those of the first)."""
    func = _at(name, basis)
    out = vmap(func)(conformers)

    assert torch.equal(out, _loop(func, conformers))
    assert not torch.allclose(out[0], out[1])


def test_vmap_structure(basis: Basis, conformer_batch: Structure) -> None:
    """A batched structure is vmapped as a pytree."""

    def func(structure: Structure) -> Tensor:
        return overlap(basis, structure)

    out = vmap(func)(conformer_batch)
    assert torch.equal(out, func(conformer_batch))
    assert torch.equal(out[2], _at("ovlp", basis)(conformer_batch.positions[2]))


def test_vmap_alphas(basis: Basis) -> None:
    """Basis parameters can be batched as well."""
    alphas = torch.stack([basis.alphas, 1.1 * basis.alphas]).detach()

    def func(alphas: Tensor) -> Tensor:
        return overlap(basis, WATER, alphas=alphas)

    assert torch.equal(vmap(func)(alphas), _loop(func, alphas))


def test_vmap_nested(basis: Basis, conformers: Tensor) -> None:
    """Nested vmap over two batch dimensions."""
    batch = torch.stack([conformers, conformers.flip(0)])
    func = _at("ovlp", basis)
    out = vmap(vmap(func))(batch)
    assert torch.equal(out[1, 0], func(conformers[2]))


@pytest.mark.parametrize("name", ["ovlp", "j"])
def test_jacrev(name: str, basis: Basis, positions: Tensor) -> None:
    """`jacrev` (vmap over the backward pass) matches autograd."""
    func = _at(name, basis)
    assert torch.allclose(jacrev(func)(positions), jacobian(func, positions))


def test_vmap_jacrev(basis: Basis, conformers: Tensor) -> None:
    """Jacobians of all conformers in one call."""
    func = _at("r0", basis)
    ref = _loop(lambda pos: jacobian(func, pos), conformers)
    assert torch.allclose(vmap(jacrev(func))(conformers), ref)


def test_vmap_grad_and_hessian(basis: Basis, conformers: Tensor) -> None:
    """Gradients and Hessians of a scalar function over conformers."""

    def func(pos: Tensor) -> Tensor:
        return _at("ovlp", basis)(pos).pow(2).sum()

    ref = _loop(lambda pos: jacobian(func, pos), conformers)
    assert torch.allclose(vmap(grad(func))(conformers), ref)

    ref = _loop(lambda pos: hessian(func, pos), conformers)
    assert torch.allclose(vmap(jacrev(grad(func)))(conformers), ref)


def test_vmap_grad_alphas(basis: Basis, conformers: Tensor) -> None:
    """Gradient w.r.t. shared exponents for every conformer."""
    alphas = basis.alphas.detach()

    def func(a: Tensor, pos: Tensor) -> Tensor:
        structure = WATER.replace(positions=pos)
        return overlap(basis, structure, alphas=a).pow(2).sum()

    def reference(pos: Tensor) -> Tensor:
        a = alphas.clone().requires_grad_(True)
        return torch.autograd.grad(func(a, pos), a)[0]

    out = vmap(grad(func), in_dims=(None, 0))(alphas, conformers)
    ref = _loop(reference, conformers)
    assert ref.abs().sum() > 0
    assert torch.allclose(out, ref)


def _gauss_newton(
    func: Callable[[Tensor], Tensor], positions: Tensor
) -> Tensor:
    """Hessian of a least-squares loss at zero residual, 2 J^T J"""
    jac = jacobian(func, positions).reshape(-1, positions.numel())
    shape = positions.shape
    return (2 * jac.T @ jac).reshape(*shape, *shape)


def test_hessian_zero_residual(basis: Basis, positions: Tensor) -> None:
    """
    At zero residual, the Hessian of a least-squares loss is 2 J^T J, which
    is entirely the derivative w.r.t. the (zero) backward-pass vector.
    """
    func = _at("ovlp", basis)
    reference = func(positions).detach()

    def loss(pos: Tensor) -> Tensor:
        return (func(pos) - reference).pow(2).sum()

    ref = _gauss_newton(func, positions)
    assert ref.abs().max() > 0.1
    assert torch.allclose(hessian(loss, positions), ref)
    assert torch.allclose(jacrev(grad(loss))(positions), ref)


# padded batches of different molecules


@pytest.mark.parametrize("name", ["ovlp", "r0", "jj", "nuc"])
def test_mixed(name: str, basis: Basis, mixed: Structure) -> None:
    """Each molecule gets its own layout; smaller ones are zero-padded."""

    def func(structure: Structure) -> Tensor:
        return int1e(name, basis, structure)

    out = vmap(func)(mixed)
    assert torch.equal(out, func(mixed))

    naos = []
    for i, molecule in enumerate(MOLECULES):
        ref = func(molecule)
        nao = ref.shape[-1]
        naos.append(nao)
        assert torch.equal(out[i, ..., :nao, :nao], ref)
        assert (out[i, ..., nao:, :] == 0).all()
        assert (out[i, ..., :, nao:] == 0).all()
    assert naos[0] < naos[1] == out.shape[-1]


def test_mixed_single_padded(basis: Basis, mixed: Structure) -> None:
    """Padding atoms of a single molecule do not change the integrals."""
    water = Structure(numbers=mixed.numbers[0], positions=mixed.positions[0])
    assert water.numbers.tolist() == [8, 1, 1, 0, 0]
    assert torch.equal(int1e("jj", basis, water), int1e("jj", basis, WATER))


def test_mixed_gradient(basis: Basis, mixed: Structure) -> None:
    """Padding atoms get zero gradients; molecules get their own gradients."""
    positions = mixed.positions.clone().requires_grad_(True)
    overlap(basis, mixed.replace(positions=positions)).pow(2).sum().backward()
    assert positions.grad is not None

    for i, molecule in enumerate(MOLECULES):
        p = molecule.positions.clone().requires_grad_(True)
        overlap(basis, molecule.replace(positions=p)).pow(2).sum().backward()
        assert p.grad is not None
        nat = len(p)
        assert torch.allclose(positions.grad[i, :nat], p.grad)
        assert (positions.grad[i, nat:] == 0).all()


def test_mixed_vmap_jacrev(basis: Basis, mixed: Structure) -> None:
    """Jacobians of different molecules in one call."""

    def func(structure: Structure, positions: Tensor) -> Tensor:
        return int1e("j", basis, structure.replace(positions=positions))

    jac = vmap(jacrev(func, argnums=1))(mixed, mixed.positions)
    for i, molecule in enumerate(MOLECULES):
        ref = jacobian(partial(func, molecule), molecule.positions)
        nao, nat = ref.shape[-3], len(molecule.numbers)
        assert torch.allclose(jac[i, :, :nao, :nao, :nat], ref)
        assert (jac[i, :, nao:] == 0).all()
        assert (jac[i, :, :, :, nat:] == 0).all()


def test_mixed_vmap_grad_alphas(basis: Basis, mixed: Structure) -> None:
    """Gradients w.r.t. the shared exponents for different molecules."""
    alphas = basis.alphas.detach()

    def func(a: Tensor, structure: Structure) -> Tensor:
        return overlap(basis, structure, alphas=a).pow(2).sum()

    out = vmap(grad(func), in_dims=(None, 0))(alphas, mixed)
    for i, molecule in enumerate(MOLECULES):
        a = alphas.clone().requires_grad_(True)
        ref = torch.autograd.grad(func(a, molecule), a)[0]
        assert torch.allclose(out[i], ref)

    # water has no carbon: zero gradient for its exponents
    carbon = slice(3, 8)  # elements sorted: H (3), C (5), O (6)
    assert (out[0, carbon] == 0).all()
    assert (out[1, carbon] != 0).all()


def test_mixed_nested(basis: Basis, mixed: Structure) -> None:
    """Nested vmap, with inner batches of molecules of different sizes."""
    # a structure has at most one batch dimension: build it inside the vmap
    numbers = torch.stack([mixed.numbers, mixed.numbers.flip(0)])
    positions = torch.stack([mixed.positions, mixed.positions.flip(0)])

    def func(n: Tensor, p: Tensor) -> Tensor:
        return overlap(basis, Structure(numbers=n, positions=p))

    out = vmap(vmap(func))(numbers, positions)
    assert torch.equal(out[1, 0], out[0, 1])
    assert torch.equal(out[1, 1], out[0, 0])
    assert torch.equal(out[0], overlap(basis, mixed))


def test_mixed_hessian_zero_residual(basis: Basis, mixed: Structure) -> None:
    """Hessians at zero residual, for different molecules in one call."""

    def func(structure: Structure, positions: Tensor) -> Tensor:
        return overlap(basis, structure.replace(positions=positions))

    def loss(structure: Structure, positions: Tensor, ref: Tensor) -> Tensor:
        return (func(structure, positions) - ref).pow(2).sum()

    references = overlap(basis, mixed).detach()
    hess = vmap(jacrev(grad(loss, argnums=1), argnums=1))(
        mixed, mixed.positions, references
    )
    for i, molecule in enumerate(MOLECULES):
        ref = _gauss_newton(partial(func, molecule), molecule.positions)
        nat = len(molecule.numbers)
        assert torch.allclose(hess[i, :nat, :, :nat], ref)
        assert (hess[i, nat:] == 0).all()
