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
Fixtures for the test structures.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.io.structure import Structure, pack_structures

from tad_libcint import Basis

from .data import DTYPE, METHANE, NUMBERS, POSITIONS, WATER, element_shells


@pytest.fixture(params=[True, False], ids=["sph", "cart"])
def basis(request: pytest.FixtureRequest) -> Basis:
    """Basis of all test elements in spherical and cartesian AOs"""
    return Basis(element_shells(), spherical=request.param)


@pytest.fixture
def positions() -> torch.Tensor:
    """Positions of the test molecule"""
    return POSITIONS.clone()


@pytest.fixture(name="conformers")
def fixture_conformers() -> torch.Tensor:
    """Three distinct geometries of the test molecule"""
    shift = 0.1 * torch.randn(
        3, 3, dtype=DTYPE, generator=torch.Generator().manual_seed(0)
    )
    return torch.stack([POSITIONS, 1.1 * POSITIONS, POSITIONS + shift])


@pytest.fixture
def conformer_batch(conformers: torch.Tensor) -> Structure:
    """The conformers as one batched structure"""
    return Structure(numbers=NUMBERS.expand(3, -1), positions=conformers)


@pytest.fixture
def mixed() -> Structure:
    """
    Padded batch of two different molecules: water (padded with two atoms of
    atomic number 0) and methane
    """
    return pack_structures([WATER, METHANE])
