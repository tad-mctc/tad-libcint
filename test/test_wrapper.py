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
Test the libcint wrapper and the integral helpers.
"""

from __future__ import annotations

import pytest
import torch

from tad_libcint import LibcintWrapper, int1e
from tad_libcint.basis import AtomCGTOBasis, CGTOBasis
from tad_libcint.interface.integrals.utils import get_integrals

DTYPE = torch.float64


def _wrapper(
    atomzs: tuple[int | float | torch.Tensor, ...] = (8, 1),
) -> LibcintWrapper:
    """Two-atom wrapper with an s and a p shell on each atom."""
    positions = [[0.0, 0.0, 0.1], [0.0, 1.1, -0.5]]
    atombases = []
    for z, pos in zip(atomzs, positions):
        bases = [
            CGTOBasis(
                0,
                torch.tensor([5.0, 1.2], dtype=DTYPE),
                torch.tensor([0.3, 0.5], dtype=DTYPE),
            ),
            CGTOBasis(
                1,
                torch.tensor([0.9], dtype=DTYPE),
                torch.tensor([1.0], dtype=DTYPE),
            ),
        ]
        atombases.append(
            AtomCGTOBasis(
                atomz=z, bases=bases, pos=torch.tensor(pos, dtype=DTYPE)
            )
        )
    return LibcintWrapper(atombases)


def test_centre_on_r_restores_centre() -> None:
    """The original centre must be restored when leaving the context."""
    w = _wrapper()
    env = w.atm_bas_env[-1]
    before = env[4:7].copy()

    r = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)
    with w.centre_on_r(r):
        assert (env[4:7] == [1.0, 2.0, 3.0]).all()

    assert (env[4:7] == before).all()


def test_fractional_charge() -> None:
    """Fractional nuclear charges must not be truncated to integers."""
    ref = int1e("nuc", _wrapper((8, 1)))

    # integral-valued floats must reproduce the integer result
    w = _wrapper((8.0, 1.0))
    assert w.fracz is True
    assert torch.allclose(int1e("nuc", w), ref)

    # a fractional charge must change the integral
    frac = int1e("nuc", _wrapper((8.5, 1.0)))
    assert not torch.allclose(frac, ref)

    # nuclear attraction is linear in the charge of each atom
    half = int1e("nuc", _wrapper((0.0, 1.0)))
    full = int1e("nuc", _wrapper((1.0, 1.0)))
    mid = int1e("nuc", _wrapper((0.5, 1.0)))
    assert torch.allclose(mid, 0.5 * (half + full))


def test_integer_charge_is_not_fractional() -> None:
    """Integer charges keep the point-nucleus model."""
    w = _wrapper((8, 1))
    assert w.fracz is False
    assert (w.atm_bas_env[0][:, 2] == 1).all()


def test_get_integrals_empty() -> None:
    """No requested integrals gives an empty result instead of an error."""
    assert get_integrals([], [_wrapper()], lambda *_: None, []) == []


def test_symmetry_s4_removed() -> None:
    """The unusable S4 symmetry is no longer advertised."""
    from tad_libcint.interface import symmetry

    assert "s4" not in symmetry.__all__
    with pytest.raises(AttributeError):
        symmetry.s4
