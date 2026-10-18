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
Shared test data: water-like and methane-like molecules and cells with s, p
and d shells, and helpers to compare with PySCF.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from tad_mctc.io.structure import Structure
from tad_mctc.kpoints import to_cartesian

from tad_libcint import Shell
from tad_libcint.basis import Layout

DTYPE = torch.float64

NUMBERS = torch.tensor([8, 1, 1])

POSITIONS = torch.tensor(
    [[0.0, 0.0, 0.2], [0.0, 1.4, -0.9], [0.1, -1.5, -0.8]], dtype=DTYPE
)

METHANE_NUMBERS = torch.tensor([6, 1, 1, 1, 1])

METHANE_POSITIONS = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [1.2, 1.2, 1.2],
        [-1.2, -1.2, 1.2],
        [-1.2, 1.2, -1.2],
        [1.2, -1.3, -1.1],
    ],
    dtype=DTYPE,
)

WATER = Structure(numbers=NUMBERS, positions=POSITIONS)

OH = Structure(numbers=NUMBERS[:2], positions=POSITIONS[:2])

METHANE = Structure(numbers=METHANE_NUMBERS, positions=METHANE_POSITIONS)

_DIMER_ORDER = [0, 3, 1, 4, 2, 5]  # interleaved: the far pairs are scattered

DIMER = Structure(
    numbers=torch.cat([NUMBERS, NUMBERS])[_DIMER_ORDER],
    positions=torch.cat(
        [POSITIONS, POSITIONS + torch.tensor([0.0, 0.0, 30.0], dtype=DTYPE)]
    )[_DIMER_ORDER],
)
"""Two water molecules beyond the screening distance of the test basis"""

LATTICE = torch.tensor(
    [[6.5, 0.0, 0.0], [0.8, 6.2, 0.0], [0.4, -0.6, 6.8]], dtype=DTYPE
)

WATER_CELL = Structure(
    numbers=torch.tensor([8, 1, 1]),
    positions=torch.tensor(
        [[0.1, 0.2, 0.3], [0.1, 1.6, -0.8], [0.2, -1.3, -0.7]], dtype=DTYPE
    ),
    lattice=LATTICE,
)
"""Water in a triclinic cell, one hydrogen outside the cell"""

OH_CELL = Structure(
    numbers=torch.tensor([8, 1]),
    positions=torch.tensor([[0.0, 0.0, 0.0], [0.4, 1.7, 0.5]], dtype=DTYPE),
    lattice=0.9 * LATTICE.flip(0),
)
"""Hydroxyl in another cell (for batches)"""

KFRAC = torch.tensor([[0.25, 0.0, 0.5], [0.5, -0.25, 0.125]], dtype=DTYPE)
"""Fractional k-points"""

KCART = to_cartesian(KFRAC, LATTICE)
"""The fractional k-points of :data:`WATER_CELL` in Cartesian coordinates"""

ShellSpec = tuple[int, list[float], list[float]]
"""Angular momentum, exponents and coefficients of a shell"""

ELEMENT_SHELLS: dict[int, list[ShellSpec]] = {
    6: [(0, [9.0, 1.9, 0.5], [0.3, 0.5, 0.4]), (1, [2.2, 0.5], [0.6, 0.5])],
    8: [
        (0, [12.0, 2.5, 0.6], [0.2, 0.5, 0.4]),
        (1, [3.0, 0.7], [0.6, 0.5]),
        (2, [1.1], [1.0]),
    ],
    1: [(0, [3.4, 0.6], [0.3, 0.7]), (1, [0.8], [1.0])],
}


def gto_norm(angmom: int, alpha: float) -> float:
    """Normalization of a primitive in libcint's convention"""
    return math.sqrt(
        2 ** (2 * angmom + 3)
        * math.factorial(angmom + 1)
        * (2 * alpha) ** (angmom + 1.5)
        / (math.factorial(2 * angmom + 2) * math.sqrt(math.pi))
    )


def make_shells(
    spec: Mapping[int, Sequence[ShellSpec]],
) -> dict[int, list[Shell]]:
    """Shells of every element from plain numbers"""
    return {
        z: [
            Shell(
                angmom=l,
                alphas=torch.tensor(a, dtype=DTYPE),
                coeffs=torch.tensor(c, dtype=DTYPE),
            )
            for l, a, c in shells
        ]
        for z, shells in spec.items()
    }


def element_shells() -> dict[int, list[Shell]]:
    """Fresh shell tensors for every element"""
    return make_shells(ELEMENT_SHELLS)


def pyscf_shells(
    mol: Any, numbers: Sequence[int]
) -> dict[int, list[ShellSpec]]:
    """
    The shells of a PySCF molecule or cell per element, with PySCF's
    normalized coefficients (PySCF places shells per atom)
    """
    per_atom: list[list[ShellSpec]] = [[] for _ in range(mol.natm)]
    for ish in range(mol.nbas):
        # pylint: disable-next=protected-access
        coeffs = mol._libcint_ctr_coeff(ish)[:, 0].tolist()
        shell = (int(mol.bas_angular(ish)), mol.bas_exp(ish).tolist(), coeffs)
        per_atom[mol.bas_atom(ish)].append(shell)

    elements: dict[int, list[ShellSpec]] = {}
    for z, shells in zip(numbers, per_atom):
        assert elements.setdefault(z, shells) == shells
    return elements


def pyscf_ao_index(layout: Layout) -> torch.Tensor:
    """
    PySCF's AO index of each of our spherical AOs: PySCF orders spherical
    p functions as (px, py, pz), libcint as (py, pz, px)
    """
    index = torch.arange(layout.nao)
    for ish in range(layout.nsh):
        if layout.shell_angmom[ish] == 1:
            first = int(layout.ao_loc[ish])
            index[first : first + 3] = first + torch.tensor([1, 2, 0])
    return index
