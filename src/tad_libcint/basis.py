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
libcint bookkeeping of one structure (cached per list of atomic numbers).
Atomic number 0 marks padding: such atoms have no shells.

Terminology:

- primitive: a single Gaussian function
- shell: a contraction of primitives with common angular momentum and centre
- AO: one angular component of a shell (e.g. three for a spherical p shell)

Contraction coefficients are used as given; include the normalization of the
primitives in them.

Spherical AOs are ordered by :math:`m = -l, \\dots, l`, which puts the p
functions in the order (py, pz, px). Cartesian AOs follow libcint's order,
e.g. (x, y, z) for p functions.

Example
-------
>>> import torch
>>> from tad_libcint import Basis, Shell
>>> dd = {"dtype": torch.double}
>>> basis = Basis({
...     1: [Shell(0, torch.tensor([3.4, 0.6], **dd), torch.tensor([0.3, 0.7], **dd))],
...     8: [
...         Shell(0, torch.tensor([12.0, 2.5], **dd), torch.tensor([0.2, 0.5], **dd)),
...         Shell(1, torch.tensor([3.0], **dd), torch.tensor([1.0], **dd)),
...     ],
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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property, lru_cache

import numpy as np
import torch
from torch import Tensor

from .library import (
    ATM_SLOTS,
    BAS_SLOTS,
    FRAC_CHARGE_NUC,
    PTR_ENV_START,
)

__all__ = ["Basis", "Layout", "Shell"]


@dataclass(frozen=True, eq=False)
class Shell:
    """
    Contracted Gaussian shell.

    Raises
    ------
    ValueError
        If the angular momentum is negative or the exponents and coefficients
        are not one-dimensional tensors of equal, non-zero length.
    """

    angmom: int
    """Angular momentum."""

    alphas: Tensor
    """Exponents of the primitives, shape ``(nprim,)``."""

    coeffs: Tensor
    """Contraction coefficients of the primitives, shape ``(nprim,)``."""

    def __post_init__(self) -> None:
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


class Basis:
    """
    Gaussian basis set: the shells of every element.

    The exponents and coefficients of all elements are concatenated (by
    atomic number, then shell, then primitive) into :attr:`alphas` and
    :attr:`coeffs`, the differentiable parameters of the integrals. Atoms of
    one element share these parameters.

    Parameters
    ----------
    shells : Mapping[int, Sequence[Shell]]
        Shells of each element, keyed by atomic number.
    spherical : bool, optional
        Use spherical (``True``) or cartesian (``False``) AOs. Defaults to
        ``True``.
    cache_size : int, optional
        Number of layouts kept in the cache. Defaults to ``256``.

    Raises
    ------
    ValueError
        If no shells are given, an atomic number is not positive, or the
        shells' tensors have different dtypes or devices.
    """

    def __init__(
        self,
        shells: Mapping[int, Sequence[Shell]],
        *,
        spherical: bool = True,
        cache_size: int = 256,
    ) -> None:
        self.shells = {int(z): tuple(shells[z]) for z in sorted(shells)}
        self.spherical = spherical

        if not any(self.shells.values()):
            raise ValueError("A basis needs at least one shell.")
        if min(self.shells) < 1:
            raise ValueError("Atomic numbers must be positive.")

        flat = [sh for z_shells in self.shells.values() for sh in z_shells]
        tensors = [t for sh in flat for t in (sh.alphas, sh.coeffs)]
        if len({(t.dtype, t.device) for t in tensors}) != 1:
            raise ValueError(
                "All exponents and coefficients must share dtype and device."
            )
        self.alphas = torch.cat([sh.alphas for sh in flat])
        """Exponents of all primitives, shape ``(nparam,)``."""
        self.coeffs = torch.cat([sh.coeffs for sh in flat])
        """Coefficients of all primitives, shape ``(nparam,)``."""

        # (angular momentum, number of primitives, first parameter) per shell
        self._element_shells: dict[int, list[tuple[int, int, int]]] = {}
        first = 0
        for z, z_shells in self.shells.items():
            self._element_shells[z] = []
            for sh in z_shells:
                self._element_shells[z].append((sh.angmom, sh.nprim, first))
                first += sh.nprim

        self._layout = lru_cache(maxsize=cache_size)(self._build_layout)

    @property
    def nparam(self) -> int:
        """Number of primitives of all elements"""
        return self.alphas.numel()

    @property
    def dtype(self) -> torch.dtype:
        """Floating point type of the basis parameters"""
        return self.alphas.dtype

    @property
    def device(self) -> torch.device:
        """Device of the basis parameters"""
        return self.alphas.device

    @cached_property
    def uncontracted(self) -> Basis:
        """
        The same basis with every primitive as a shell of its own.

        The primitives keep their order, so :attr:`alphas` and :attr:`coeffs`
        apply to both bases.
        """
        return Basis(
            {
                z: [
                    Shell(sh.angmom, sh.alphas[i : i + 1], sh.coeffs[i : i + 1])
                    for sh in z_shells
                    for i in range(sh.nprim)
                ]
                for z, z_shells in self.shells.items()
            },
            spherical=self.spherical,
        )

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
        return self._layout(tuple(int(z) for z in numbers.tolist()))

    def _build_layout(self, numbers: tuple[int, ...]) -> Layout:
        missing = sorted(set(numbers) - set(self._element_shells) - {0})
        if missing:
            raise ValueError(f"The basis has no shells for elements {missing}.")

        atom: list[int] = []
        angmom: list[int] = []
        nprim: list[int] = []
        prim_param: list[int] = []
        for iat, z in enumerate(numbers):
            for l, n, first in self._element_shells.get(z, []):  # 0: padding
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
            spherical=self.spherical,
            device=self.device,
        )

    def __repr__(self) -> str:
        nsh = sum(len(z_shells) for z_shells in self.shells.values())
        return (
            f"{type(self).__name__}(elements={list(self.shells)}, nsh={nsh}, "
            f"spherical={self.spherical})"
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
    def ao_to_param(self) -> Tensor:
        """Basis parameter of the first primitive of each AO's shell"""
        return self._per_ao(self.prim_param[self.shell_first_prim])

    def _per_ao(self, per_shell: np.ndarray) -> Tensor:
        per_ao = np.repeat(per_shell, self.shell_nao)
        return torch.as_tensor(per_ao, dtype=torch.long, device=self.device)

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
