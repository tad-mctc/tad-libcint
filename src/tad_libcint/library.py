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
Library
=======

The boundary to the compiled *libcint* code. Everything that touches
:mod:`ctypes` lives here: lazy loading of the shared objects, the slot layout
constants of libcint's ``atm``/``bas``/``env`` arrays, and the one raw driver
call that evaluates a two-centre integral. Nothing above this module handles
C pointers.

The shared objects are expected next to this module and are loaded when the
first symbol is accessed.

Example
-------
>>> from tad_libcint.library import CINT
>>> CINT.int1e_ovlp_sph.__name__
'int1e_ovlp_sph'
"""

from __future__ import annotations

import ctypes
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "ATM_SLOTS",
    "BAS_SLOTS",
    "CGTO",
    "CINT",
    "FRAC_CHARGE_NUC",
    "LazySharedLibrary",
    "PTR_ENV_START",
    "has_symbol",
    "int2c",
]


# Slot layout of libcint's input arrays (see libcint's ``cint.h``).

PTR_ENV_START = 20
"""Number of reserved global slots at the start of ``env``."""

ATM_SLOTS = 6
"""Number of integer slots per atom in ``atm``."""

BAS_SLOTS = 8
"""Number of integer slots per shell in ``bas``."""

FRAC_CHARGE_NUC = 3
"""Nuclear model for (fractional) point charges read from ``env``."""


class LazySharedLibrary:
    """Shared object that is only loaded once one of its symbols is accessed"""

    def __init__(self, name: str, directory: Path) -> None:
        self.path = directory / f"lib{name}.so"
        self._lib: ctypes.CDLL | None = None

    def __getattr__(self, symbol: str) -> Any:
        if self._lib is None:
            self._lib = ctypes.cdll.LoadLibrary(str(self.path))
        return getattr(self._lib, symbol)


_HERE = Path(__file__).parent.resolve()

# The libraries carry a suffix to avoid clashing with PySCF's libraries of the
# same name (see `TAD_LIBRARY_SUFFIX` in libs/CMakeLists.txt).

CINT = LazySharedLibrary("cint_tad", _HERE)
"""The *libcint* integral library."""

CGTO = LazySharedLibrary("cgto_tad", _HERE)
"""PySCF's ``libcgto`` with the drivers that fill whole integral matrices."""


def has_symbol(symbol: str) -> bool:
    """Check whether *libcint* exports the given symbol"""
    try:
        getattr(CINT, symbol)
    except AttributeError:
        return False
    return True


def _ptr(array: np.ndarray) -> ctypes.c_void_p:
    return array.ctypes.data_as(ctypes.c_void_p)


def int2c(
    symbol: str,
    comp_shape: tuple[int, ...],
    hermitian: bool,
    *,
    shls_slice: tuple[int, int, int, int],
    pairs: np.ndarray | None = None,
    components: Sequence[int] | None = None,
    atm: np.ndarray,
    bas: np.ndarray,
    env: np.ndarray,
    ao_loc: np.ndarray,
) -> np.ndarray:
    """
    Evaluate a block of a two-centre integral with libcint.

    Parameters
    ----------
    symbol : str
        Name of the libcint function, e.g. ``"int1e_ovlp_sph"``.
    comp_shape : tuple[int, ...]
        Shape of the integral's components, e.g. ``(3,)`` for a dipole.
    hermitian : bool
        Only compute the lower triangle and mirror it. Must only be set for
        integrals that are symmetric under exchange of bra and ket, and for
        blocks with the same bra and ket shells.
    shls_slice : tuple[int, int, int, int]
        First and last (exclusive) shell of the bra and of the ket.
    pairs : np.ndarray | None, optional
        Bra and ket shells of the shell pairs to evaluate, shape
        ``(npairs, 2)``. The elements of all other shell pairs are zero.
        Defaults to all shell pairs of the slice.
    components : Sequence[int] | None, optional
        Flat indices of the components to keep, in this order. The others
        are never stored, not even temporarily for the whole block. Defaults
        to all components.
    atm : np.ndarray
        Atom slots, shape ``(nat, ATM_SLOTS)``, ``int32``.
    bas : np.ndarray
        Shell slots, shape ``(nsh, BAS_SLOTS)``, ``int32``.
    env : np.ndarray
        Floating point environment, ``float64``.
    ao_loc : np.ndarray
        Offset of each shell's first AO, shape ``(nsh + 1,)``, ``int32``.

    Returns
    -------
    np.ndarray
        Integral of shape ``(*comp_shape, nao_bra, nao_ket)``, or
        ``(len(components), nao_bra, nao_ket)``.
    """
    atm = np.ascontiguousarray(atm, dtype=np.int32)
    bas = np.ascontiguousarray(bas, dtype=np.int32)
    env = np.ascontiguousarray(env, dtype=np.float64)
    ao_loc = np.ascontiguousarray(ao_loc, dtype=np.int32)

    nat, nsh = len(atm), len(bas)
    ish0, ish1, jsh0, jsh1 = shls_slice
    nao_bra = int(ao_loc[ish1] - ao_loc[ish0])
    nao_ket = int(ao_loc[jsh1] - ao_loc[jsh0])
    ncomp = int(np.prod(comp_shape))
    if components is None:
        comp_map, out_shape = None, comp_shape
    else:
        comp_map = np.asarray(components, dtype=np.int32)
        out_shape = (len(comp_map),)
        if ((comp_map < 0) | (comp_map >= ncomp)).any():
            raise ValueError(f"Components {components} out of {ncomp}.")

    # libcint fills each component in column-major order
    shape = (math.prod(out_shape), nao_ket, nao_bra)
    out = np.empty(shape) if pairs is None else np.zeros(shape)

    # optimizer with precomputed, geometry-dependent data (must not outlive
    # the `env` it was created from)
    opt = ctypes.c_void_p()
    optimizer = getattr(CINT, symbol.rsplit("_", 1)[0] + "_optimizer")
    optimizer(ctypes.byref(opt), _ptr(atm), nat, _ptr(bas), nsh, _ptr(env))

    block = (
        getattr(CINT, symbol),
        _ptr(out),
        ncomp,
        None if comp_map is None else _ptr(comp_map),
        0 if comp_map is None else len(comp_map),
        int(hermitian),
        (ctypes.c_int * 4)(*shls_slice),
    )
    system = (_ptr(ao_loc), opt, _ptr(atm), nat, _ptr(bas), nsh, _ptr(env))
    try:
        if pairs is None:
            CGTO.GTOint2c(*block, *system)
        else:
            pairs = np.ascontiguousarray(pairs, dtype=np.int32)
            CGTO.GTOint2c_pairs(*block, _ptr(pairs), len(pairs), *system)
    finally:
        CINT.CINTdel_optimizer(ctypes.byref(opt))

    return out.swapaxes(-1, -2).reshape(*out_shape, nao_bra, nao_ket)
