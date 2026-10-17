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
Test the parsing and manipulation of integral names.
"""

from __future__ import annotations

import pytest

from tad_libcint.names import IntegralName


@pytest.mark.parametrize(
    "shortname, bra, op, ket",
    [
        ("ovlp", (), "ovlp", ()),
        ("ipovlp", ("ip",), "ovlp", ()),
        ("ipipovlpip", ("ip", "ip"), "ovlp", ("ip",)),
        ("rrr0r0rr", ("rr",), "r0r0", ("rr",)),
        ("ipjj", ("ip",), "jj", ()),
        ("r0r0r0", (), "r0r0r0", ()),
    ],
)
def test_parse_roundtrip(
    shortname: str, bra: tuple[str, ...], op: str, ket: tuple[str, ...]
) -> None:
    """Names split into bra operators, central operator and ket operators."""
    name = IntegralName.parse(shortname)
    assert (name.bra, name.op, name.ket) == (bra, op, ket)
    assert name.shortname == shortname


@pytest.mark.parametrize("shortname", ["kin", "ipfoo", "", "ovlpx", "ipip"])
def test_parse_invalid(shortname: str) -> None:
    """Unknown operators are rejected with a helpful message."""
    with pytest.raises(ValueError, match="Cannot parse"):
        IntegralName.parse(shortname)


def test_invalid_operator() -> None:
    """Direct construction validates the operators as well."""
    with pytest.raises(ValueError, match="Unsupported operator"):
        IntegralName("kin")
    with pytest.raises(ValueError, match="Unknown derivative"):
        IntegralName("ovlp", bra=("xx",))


def test_comp_shape() -> None:
    """Components are ordered bra, central operator, ket."""
    assert IntegralName.parse("ovlp").comp_shape == ()
    assert IntegralName.parse("ipr0r0ip").comp_shape == (3, 9, 3)
    assert IntegralName.parse("rrovlp").comp_shape == ()


def test_packed() -> None:
    """Packed quadrupoles keep 6 of the 9 operator components."""
    name = IntegralName("r0r0", bra=("ip",), packed=True)
    assert name.comp_shape == (3, 6)
    assert name.unpacked.comp_shape == (3, 9)
    assert name.components == [
        9 * b + c for b in range(3) for c in (0, 3, 4, 6, 7, 8)
    ]
    assert name.differentiate("ip", "ket").components is not None
    assert name.symbol(spherical=True) == "int1e_ipr0r0_sph"
    assert IntegralName.parse("jj").components is None
    with pytest.raises(ValueError, match="Only quadrupoles"):
        IntegralName("j", packed=True)


def test_differentiate_and_transpose() -> None:
    """Derivatives are prepended; transposition swaps bra and ket."""
    name = IntegralName.parse("ipr0")
    assert name.differentiate("ip", "bra").shortname == "ipipr0"
    assert name.differentiate("ip", "ket").shortname == "ipr0ip"
    assert name.transposed().shortname == "r0ip"
    assert name.symbol(spherical=False) == "int1e_ipr0_cart"


def test_symmetry() -> None:
    """Ket-centred operators are neither symmetric nor hermitian."""
    assert IntegralName.parse("ovlp").is_hermitian
    assert IntegralName.parse("r0r0").is_hermitian
    assert not IntegralName.parse("ipovlp").is_hermitian
    assert not IntegralName.parse("j").is_symmetric
    assert not IntegralName.parse("jj").is_hermitian
