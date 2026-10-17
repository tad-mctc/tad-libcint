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
Basis
=====

A :class:`Basis` assigns contracted Gaussian shells (:class:`Shell`) to
elements. It holds the primitive exponents and contraction coefficients of
all elements, but no structure: the atomic numbers and positions are passed
to every integral call. One basis therefore serves conformers as well as
different structures, including padded batches of them.

For a given list of atomic numbers, the basis builds a :class:`Layout`, the
libcint bookkeeping of one structure. Layouts are cached by the shells'
angular momenta and numbers of primitives, so copies of a basis (e.g. by
``.to()`` or inside ``torch.func`` transforms) share them. Atomic number 0
marks padding: such atoms have no shells.

:class:`Shell` and :class:`Basis` are :class:`~tad_mctc.tree.Node` objects:
their tensors (exponents and coefficients) are pytree leaves, so a basis
converts with ``.to()`` and passes through ``torch.func`` transforms.

Terminology:

- primitive: a single Gaussian function
- shell: a contraction of primitives with common angular momentum and centre
- AO: one angular component of a shell (e.g. three for a spherical p shell)

Contraction coefficients are used as given. A spherical AO is the radial part
:math:`\\sum_k c_k r^l e^{-\\alpha_k r^2}` times a real spherical harmonic
with unit norm on the sphere (libcint's convention). Coefficients of other
conventions differ by a factor per angular momentum, e.g. by
:math:`\\sqrt{4\\pi / (2l + 1)}` from coefficients that normalize the
Cartesian function :math:`x^l e^{-\\alpha r^2}`. :meth:`Basis.normalized`
scales the coefficients of every shell to unit self-overlap.

Spherical AOs are ordered by :math:`m = -l, \\dots, l`, which puts the p
functions in the order (py, pz, px). Cartesian AOs follow libcint's order,
e.g. (x, y, z) for p functions.

Example
-------
>>> import torch
>>> from tad_libcint import Basis, Shell
>>> def shell(l, alphas, coeffs):
...     dd = {"dtype": torch.double}
...     return Shell(
...         angmom=l,
...         alphas=torch.tensor(alphas, **dd),
...         coeffs=torch.tensor(coeffs, **dd),
...     )
>>> basis = Basis(shells={
...     1: [shell(0, [3.4, 0.6], [0.3, 0.7])],
...     8: [shell(0, [12.0, 2.5], [0.2, 0.5]), shell(1, [3.0], [1.0])],
... })
>>> basis
Basis(elements=[1, 8], nsh=3, spherical=True)
>>> layout = basis.layout(torch.tensor([8, 1, 1, 0]))
>>> layout
Layout(nat=4, nsh=4, nao=6, spherical=True)
>>> layout.ao_to_atom
tensor([0, 0, 0, 0, 1, 2])
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any

import numpy as np
import torch
from tad_mctc.tree import Node, child, context
from torch import Tensor

from .library import (
    ATM_SLOTS,
    BAS_SLOTS,
    FRAC_CHARGE_NUC,
    PTR_ENV_START,
)

__all__ = ["Basis", "Layout", "Shell"]


class Shell(Node):
    """
    Contracted Gaussian shell.

    Raises
    ------
    ValueError
        If the angular momentum is negative or the exponents and coefficients
        are not one-dimensional tensors of equal, non-zero length.
    """

    angmom: int = context()
    """Angular momentum."""

    alphas: Tensor = child()
    """Exponents of the primitives, shape ``(nprim,)``."""

    coeffs: Tensor = child()
    """Contraction coefficients of the primitives, shape ``(nprim,)``."""

    def _validate(self) -> None:
        if self.angmom < 0:
            raise ValueError(f"Negative angular momentum {self.angmom}.")
        if self.alphas.ndim != 1 or self.alphas.numel() == 0:
            raise ValueError(
                "Exponents must be a non-empty 1D tensor, got shape "
                f"{tuple(self.alphas.shape)}."
            )
        if self.alphas.shape != self.coeffs.shape:
            raise ValueError(
                f"Shapes of exponents {tuple(self.alphas.shape)} and "
                f"coefficients {tuple(self.coeffs.shape)} differ."
            )

    @property
    def nprim(self) -> int:
        """Number of primitives"""
        return self.alphas.numel()

    def normalized(self) -> Shell:
        r"""
        The shell with its coefficients scaled to unit self-overlap.

        The self-overlap of the radial part
        :math:`\sum_k c_k r^l e^{-\alpha_k r^2}` is

        .. math::

            \sum_{k,k'} c_k c_{k'}
            \frac{\Gamma(l + 3/2)}{2 (\alpha_k + \alpha_{k'})^{l + 3/2}}.

        The scaling is differentiable, so gradients w.r.t. the original
        exponents and coefficients include the normalization.

        Returns
        -------
        Shell
            The normalized shell.
        """
        l = self.angmom
        p = self.alphas.unsqueeze(-1) + self.alphas.unsqueeze(-2)
        cc = self.coeffs.unsqueeze(-1) * self.coeffs.unsqueeze(-2)
        overlap = 0.5 * math.gamma(l + 1.5) * (cc * p ** -(l + 1.5)).sum()
        return self.replace(coeffs=self.coeffs * overlap.rsqrt())


ElementShells = tuple[tuple[int, tuple[tuple[int, int, int], ...]], ...]
"""
Static structure of a basis: for each element (sorted by atomic number), the
angular momentum, number of primitives and first parameter of each shell.
"""


class Basis(Node):
    """
    Gaussian basis set: the shells of every element.

    The exponents and coefficients of all elements are concatenated (by
    atomic number, then shell, then primitive) into :attr:`alphas` and
    :attr:`coeffs`, the differentiable parameters of the integrals. Atoms of
    one element share these parameters.

    Raises
    ------
    ValueError
        If no shells are given or an atomic number is not positive.
    TypeError
        If the shells' tensors have different dtypes.
    RuntimeError
        If the shells' tensors are on different devices.
    """

    shells: Mapping[int, Sequence[Shell]] = child()
    """Shells of each element, keyed by atomic number."""

    spherical: bool = context(default=True)
    """Use spherical (``True``) or cartesian (``False``) AOs."""

    def _normalize(self) -> dict[str, Any]:
        # sorted by atomic number, which orders the parameters
        items = sorted(self.shells.items())
        shells = {int(z): tuple(z_shells) for z, z_shells in items}
        return {"shells": shells}

    def _validate(self) -> None:
        if not any(self.shells.values()):
            raise ValueError("A basis needs at least one shell.")
        if min(self.shells) < 1:
            raise ValueError("Atomic numbers must be positive.")

    @cached_property
    def alphas(self) -> Tensor:
        """Exponents of all primitives, shape ``(nparam,)``"""
        return torch.cat([sh.alphas for sh in self._flat_shells])

    @cached_property
    def coeffs(self) -> Tensor:
        """Coefficients of all primitives, shape ``(nparam,)``"""
        return torch.cat([sh.coeffs for sh in self._flat_shells])

    @property
    def _flat_shells(self) -> list[Shell]:
        return [sh for z_shells in self.shells.values() for sh in z_shells]

    @property
    def nparam(self) -> int:
        """Number of primitives of all elements"""
        return sum(sh.nprim for sh in self._flat_shells)

    @cached_property
    def element_shells(self) -> ElementShells:
        """The static structure of the basis (see :data:`ElementShells`)"""
        structure = []
        first = 0
        for z, z_shells in self.shells.items():
            shells = []
            for sh in z_shells:
                shells.append((sh.angmom, sh.nprim, first))
                first += sh.nprim
            structure.append((z, tuple(shells)))
        return tuple(structure)

    @cached_property
    def uncontracted(self) -> Basis:
        """
        The same basis with every primitive as a shell of its own.

        The primitives keep their order, so :attr:`alphas` and :attr:`coeffs`
        apply to both bases.
        """
        shells = {
            z: [
                Shell(
                    angmom=sh.angmom,
                    alphas=sh.alphas[i : i + 1],
                    coeffs=sh.coeffs[i : i + 1],
                )
                for sh in z_shells
                for i in range(sh.nprim)
            ]
            for z, z_shells in self.shells.items()
        }
        return Basis(shells=shells, spherical=self.spherical)

    def normalized(self) -> Basis:
        """
        The basis with every shell scaled to unit self-overlap (see
        :meth:`Shell.normalized`).

        Spherical AOs then have unit norm, Cartesian AOs only up to p
        functions (e.g. ``xy`` and ``xx`` differ in norm).

        Returns
        -------
        Basis
            The normalized basis.
        """
        shells = {
            z: [sh.normalized() for sh in z_shells]
            for z, z_shells in self.shells.items()
        }
        return self.replace(shells=shells)

    def layout(self, numbers: Tensor) -> Layout:
        """
        Layout of the structure with the given atomic numbers.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers, shape ``(nat,)``. Zeros mark padding.

        Returns
        -------
        Layout
            The (cached) layout.

        Raises
        ------
        ValueError
            If the basis has no shells for one of the elements.
        """
        if numbers.ndim != 1:
            raise ValueError(
                f"Expected atomic numbers of shape (nat,), got "
                f"{tuple(numbers.shape)}."
            )
        return _layout(
            self.element_shells,
            self.spherical,
            tuple(numbers.tolist()),
            self.device,
        )

    def __repr__(self) -> str:
        nsh = sum(len(z_shells) for z_shells in self.shells.values())
        return (
            f"{type(self).__name__}(elements={list(self.shells)}, nsh={nsh}, "
            f"spherical={self.spherical})"
        )


@lru_cache(maxsize=256)
def _layout(
    element_shells: ElementShells,
    spherical: bool,
    numbers: tuple[int, ...],
    device: torch.device,
) -> Layout:
    """The layout of a structure (see :meth:`Basis.layout`)"""
    shells = dict(element_shells)
    missing = sorted(set(numbers) - set(shells) - {0})
    if missing:
        raise ValueError(f"The basis has no shells for elements {missing}.")

    atom: list[int] = []
    angmom: list[int] = []
    nprim: list[int] = []
    prim_param: list[int] = []
    for iat, z in enumerate(numbers):
        for l, n, first in shells.get(z, ()):  # 0: padding
            atom.append(iat)
            angmom.append(l)
            nprim.append(n)
            prim_param.extend(range(first, first + n))

    return Layout(
        nat=len(numbers),
        shell_atom=np.array(atom, dtype=np.int64),
        shell_angmom=np.array(angmom, dtype=np.int64),
        shell_nprim=np.array(nprim, dtype=np.int64),
        prim_param=np.array(prim_param, dtype=np.int64),
        spherical=spherical,
        device=device,
    )


@dataclass(frozen=True, eq=False, repr=False)
class Layout:
    """
    libcint bookkeeping of one structure (see :meth:`Basis.layout`).

    Shells are ordered by atom, then by the element's shell order.
    """

    nat: int
    """Number of atoms, including padding."""

    shell_atom: np.ndarray
    """Atom index of each shell, shape ``(nsh,)``."""

    shell_angmom: np.ndarray
    """Angular momentum of each shell, shape ``(nsh,)``."""

    shell_nprim: np.ndarray
    """Number of primitives of each shell, shape ``(nsh,)``."""

    prim_param: np.ndarray
    """Index of each primitive in the basis parameters, shape ``(nprim,)``."""

    spherical: bool
    """Spherical (``True``) or cartesian (``False``) AOs."""

    device: torch.device
    """Device of the index tensors."""

    # dimensions

    @property
    def nsh(self) -> int:
        """Number of shells"""
        return len(self.shell_atom)

    @property
    def nprim(self) -> int:
        """Number of primitives"""
        return len(self.prim_param)

    @property
    def nao(self) -> int:
        """Number of atomic orbitals"""
        return int(self.ao_loc[-1])

    # AO bookkeeping

    @cached_property
    def shell_nao(self) -> np.ndarray:
        """Number of AOs of each shell, shape ``(nsh,)``"""
        l = self.shell_angmom
        return 2 * l + 1 if self.spherical else (l + 1) * (l + 2) // 2

    @cached_property
    def ao_loc(self) -> np.ndarray:
        """Index of the first AO of each shell and ``nao``, ``(nsh + 1,)``"""
        return np.concatenate([[0], np.cumsum(self.shell_nao)]).astype(np.int32)

    @cached_property
    def ao_to_atom(self) -> Tensor:
        """Atom index of each AO, shape ``(nao,)``"""
        return self._per_ao(self.shell_atom)

    @cached_property
    def ao_to_shell(self) -> Tensor:
        """Shell index of each AO, shape ``(nao,)``"""
        return self._per_ao(np.arange(self.nsh))

    @cached_property
    def shell_first_prim(self) -> np.ndarray:
        """Index of the first primitive of each shell, shape ``(nsh,)``"""
        return np.cumsum(self.shell_nprim) - self.shell_nprim

    @cached_property
    def prim_loc(self) -> np.ndarray:
        """Index of the first primitive of each shell and ``nprim``"""
        return np.concatenate([self.shell_first_prim, [self.nprim]])

    @cached_property
    def ao_to_param(self) -> Tensor:
        """Basis parameter of the first primitive of each AO's shell"""
        return self._per_ao(self.prim_param[self.shell_first_prim])

    def _per_ao(self, per_shell: np.ndarray) -> Tensor:
        per_ao = np.repeat(per_shell, self.shell_nao)
        return torch.as_tensor(per_ao, dtype=torch.long, device=self.device)

    def aos(self, shells: range) -> slice:
        """AOs of consecutive shells"""
        return slice(
            int(self.ao_loc[shells.start]), int(self.ao_loc[shells.stop])
        )

    def shell_pairs(self, atom_pairs: np.ndarray) -> np.ndarray:
        """
        Pairs of a shell of the first and a shell of the second atom of each
        atom pair, shape ``(npairs, 2)``
        """
        nshell = np.bincount(self.shell_atom, minlength=self.nat)
        return _member_pairs(atom_pairs, nshell)

    def uncontracted_pairs(self, shell_pairs: np.ndarray) -> np.ndarray:
        """
        Pairs of shells of :attr:`Basis.uncontracted` (one per primitive) that
        make up the given shell pairs, shape ``(npairs, 2)``
        """
        return _member_pairs(shell_pairs, self.shell_nprim)

    def uncontracted_shells(self, shells: range) -> range:
        """
        Shells of :attr:`Basis.uncontracted` (one per primitive) that make up
        consecutive shells of this layout
        """
        return range(
            int(self.prim_loc[shells.start]), int(self.prim_loc[shells.stop])
        )

    @cached_property
    def uao_to_ao(self) -> Tensor:
        """
        Contracted AO of each AO of the uncontracted structure, i.e., of the
        layout of :attr:`Basis.uncontracted` for the same atoms
        """
        uao_to_ao = [
            ao
            for ish in range(self.nsh)
            for _ in range(self.shell_nprim[ish])
            for ao in range(self.ao_loc[ish], self.ao_loc[ish + 1])
        ]
        return torch.tensor(uao_to_ao, dtype=torch.long, device=self.device)

    # libcint input arrays

    # Layout of `env` after libcint's reserved slots:
    # [ coordinates (3 * nat) | zeta, charge (2 * nat) | alphas | coeffs ]

    @cached_property
    def bas(self) -> np.ndarray:
        """libcint's shell slots, shape ``(nsh, 8)``"""
        ptr_exp = PTR_ENV_START + 5 * self.nat + self.shell_first_prim
        bas = np.zeros((self.nsh, BAS_SLOTS), dtype=np.int32)
        bas[:, 0] = self.shell_atom
        bas[:, 1] = self.shell_angmom
        bas[:, 2] = self.shell_nprim
        bas[:, 3] = 1  # number of contractions
        bas[:, 5] = ptr_exp
        bas[:, 6] = ptr_exp + self.nprim
        return bas

    def atm(self, charges: Tensor) -> np.ndarray:
        """
        libcint's atom slots, shape ``(nat, 6)``.

        All atoms use libcint's fractional nuclear model, which reads the
        charges from `env` as they are (the point model takes the absolute
        value of an integer charge).
        """
        ptr_nuclear = PTR_ENV_START + 3 * self.nat + 2 * np.arange(self.nat)
        atm = np.zeros((self.nat, ATM_SLOTS), dtype=np.int32)
        atm[:, 0] = _to_numpy(charges).astype(np.int32)
        atm[:, 1] = PTR_ENV_START + 3 * np.arange(self.nat)
        atm[:, 2] = FRAC_CHARGE_NUC
        atm[:, 3] = ptr_nuclear  # Gaussian nuclear model (unused)
        atm[:, 4] = ptr_nuclear + 1  # fractional charge
        return atm

    def env(
        self, charges: Tensor, positions: Tensor, alphas: Tensor, coeffs: Tensor
    ) -> np.ndarray:
        """
        Build libcint's floating point environment for one geometry.

        Parameters
        ----------
        charges : Tensor
            Nuclear charges, shape ``(nat,)``.
        positions : Tensor
            Cartesian coordinates in Bohr, shape ``(nat, 3)``.
        alphas : Tensor
            Exponents of the basis, shape ``(nparam,)``.
        coeffs : Tensor
            Coefficients of the basis, shape ``(nparam,)``.

        Returns
        -------
        np.ndarray
            The environment, ``float64``.
        """
        nuclear = np.stack([np.zeros(self.nat), _to_numpy(charges)], axis=-1)
        return np.concatenate(
            [
                np.zeros(PTR_ENV_START),
                _to_numpy(positions).reshape(-1),
                nuclear.reshape(-1),
                _to_numpy(alphas)[self.prim_param],
                _to_numpy(coeffs)[self.prim_param],
            ]
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(nat={self.nat}, nsh={self.nsh}, "
            f"nao={self.nao}, spherical={self.spherical})"
        )


def _to_numpy(x: Tensor) -> np.ndarray:
    return x.detach().to(device="cpu", dtype=torch.float64).numpy()


def _member_pairs(pairs: np.ndarray, size: np.ndarray) -> np.ndarray:
    """
    All pairs of members of the groups in each pair of groups, where the
    groups consist of consecutive members (`size` of each group)
    """
    first = np.cumsum(size) - size
    bra, ket = pairs[:, 0], pairs[:, 1]

    # for each member pair: its pair of groups and its index within them
    npairs = size[bra] * size[ket]
    group = np.repeat(np.arange(len(pairs)), npairs)
    index = np.arange(npairs.sum()) - np.repeat(
        np.cumsum(npairs) - npairs, npairs
    )

    nket = size[ket][group]
    return np.stack(
        [first[bra][group] + index // nket, first[ket][group] + index % nket],
        axis=-1,
    )
