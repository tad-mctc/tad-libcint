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
Shared test data: water-like and methane-like molecules with s, p and d
shells.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from tad_mctc.io.structure import Structure

from tad_libcint import Shell

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
