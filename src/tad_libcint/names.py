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
r"""
Integral Names
==============

libcint addresses every integral by name. A one-electron integral name has
three parts: operators acting on the bra function, the central operator, and
operators acting on the ket function. For example, ``"ipovlpip"`` is
:math:`\langle \nabla \mu | \nabla \nu \rangle`, i.e., ``ip`` on the
bra, ``ovlp`` in the middle and ``ip`` on the ket.

The derivatives required for the backward pass are obtained by prepending a
derivative operator to the bra or ket:

- ``ip``: nabla with respect to the electron coordinate (adds an axis of 3)
- ``rr``: :math:`r^2` relative to the function's centre (the derivative with
  respect to the Gaussian exponent, up to a sign; adds no axis)

Example
-------
>>> from tad_libcint.names import IntegralName
>>> name = IntegralName.parse("r0")
>>> name.differentiate("ip", "ket").shortname
'r0ip'
>>> name.differentiate("ip", "ket").comp_shape
(3, 3)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Literal

__all__ = [
    "IntegralName",
    "Side",
    "CENTRAL_OPERATORS",
    "DERIVATIVE_OPERATORS",
    "KET_CENTRED_OPERATORS",
]

Side = Literal["bra", "ket"]

DERIVATIVE_OPERATORS: dict[str, tuple[int, ...]] = {"ip": (3,), "rr": ()}
"""Operators acting on a basis function and the axes they add."""

CENTRAL_OPERATORS: dict[str, tuple[int, ...]] = {
    "ovlp": (),
    "nuc": (),
    "r0": (3,),
    "r0r0": (9,),
    "r0r0r0": (27,),
    "j": (3,),
    "jj": (9,),
    "jjj": (27,),
}
"""Supported central operators and the shape of their components."""

KET_CENTRED_OPERATORS: dict[str, str] = {"j": "ovlp", "jj": "j", "jjj": "jj"}
r"""
Operators :math:`(\mathbf{r} - \mathbf{R}_\nu)^{\otimes n}` that are
centred on the ket function, mapped to the operator of rank ``n - 1``. They
depend on the ket's position themselves and are not bra/ket symmetric.
"""

_PATTERN = re.compile(
    "^((?:{d})*)({c})((?:{d})*)$".format(
        d="|".join(DERIVATIVE_OPERATORS),
        c="|".join(sorted(CENTRAL_OPERATORS, key=len, reverse=True)),
    )
)
_TOKEN = re.compile("|".join(DERIVATIVE_OPERATORS))


@dataclass(frozen=True)
class IntegralName:
    """
    Structured name of a two-centre one-electron integral.

    Raises
    ------
    ValueError
        If a name cannot be parsed or uses an unsupported operator.
    """

    op: str
    """Central operator, e.g. ``"ovlp"``."""

    bra: tuple[str, ...] = ()
    """Derivative operators on the bra function, outermost first."""

    ket: tuple[str, ...] = ()
    """Derivative operators on the ket function, outermost first."""

    def __post_init__(self) -> None:
        if self.op not in CENTRAL_OPERATORS:
            raise ValueError(
                f"Unsupported operator '{self.op}'. Supported operators are: "
                f"{list(CENTRAL_OPERATORS)}."
            )
        for op in self.bra + self.ket:
            if op not in DERIVATIVE_OPERATORS:
                raise ValueError(f"Unknown derivative operator '{op}'.")

    @classmethod
    def parse(cls, shortname: str) -> IntegralName:
        """Split a short name such as ``"ipovlp"`` into its parts"""
        match = _PATTERN.match(shortname)
        if match is None:
            raise ValueError(
                f"Cannot parse integral name '{shortname}'. Expected "
                "'<bra ops><operator><ket ops>' with operators from "
                f"{list(CENTRAL_OPERATORS)} and derivative operators from "
                f"{list(DERIVATIVE_OPERATORS)}."
            )
        bra, op, ket = match.groups()
        return cls(op, tuple(_TOKEN.findall(bra)), tuple(_TOKEN.findall(ket)))

    @property
    def shortname(self) -> str:
        """Name without the integral type, e.g. ``"ipovlp"``"""
        return "".join(self.bra) + self.op + "".join(self.ket)

    def symbol(self, spherical: bool) -> str:
        """Name of the corresponding libcint function"""
        suffix = "sph" if spherical else "cart"
        return f"int1e_{self.shortname}_{suffix}"

    @property
    def bra_shape(self) -> tuple[int, ...]:
        """Component axes added by the bra operators"""
        return sum((DERIVATIVE_OPERATORS[op] for op in self.bra), ())

    @property
    def op_shape(self) -> tuple[int, ...]:
        """Component axes of the central operator"""
        return CENTRAL_OPERATORS[self.op]

    @property
    def ket_shape(self) -> tuple[int, ...]:
        """Component axes added by the ket operators"""
        return sum((DERIVATIVE_OPERATORS[op] for op in self.ket), ())

    @property
    def comp_shape(self) -> tuple[int, ...]:
        """Shape of all components, in libcint's order (bra, op, ket)"""
        return self.bra_shape + self.op_shape + self.ket_shape

    @property
    def is_symmetric(self) -> bool:
        """Whether the central operator is symmetric under bra/ket exchange"""
        return self.op not in KET_CENTRED_OPERATORS

    @property
    def is_hermitian(self) -> bool:
        """Whether the integral matrix is symmetric in its AO indices"""
        return self.is_symmetric and not self.bra and not self.ket

    def differentiate(self, deriv: str, side: Side) -> IntegralName:
        """Integral with `deriv` applied (outermost) to the bra or ket"""
        if side == "bra":
            return replace(self, bra=(deriv, *self.bra))
        return replace(self, ket=(deriv, *self.ket))

    def transposed(self) -> IntegralName:
        """Integral with bra and ket operators exchanged"""
        return replace(self, bra=self.ket, ket=self.bra)

    def __str__(self) -> str:
        return self.shortname
