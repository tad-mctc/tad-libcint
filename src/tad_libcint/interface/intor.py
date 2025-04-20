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
Interface: Integrals
====================

This module contains the integral functions.
"""
from __future__ import annotations

import ctypes
import operator
from functools import reduce

import numpy as np
from tad_mctc.convert import numpy_to_tensor

from tad_libcint.api import CGTO, CINT
from tad_libcint.typing import Tensor

from .namemanager import IntorNameManager
from .utils import int2ctypes, np2ctypes
from .wrapper import LibcintWrapper

################### integrator (direct interface to libcint) ###################


class Intor:
    """
    Interface to the libcint integrals.
    """

    def __init__(
        self,
        int_nmgr: IntorNameManager,
        wrappers: list[LibcintWrapper],
        hermitian: bool = False,
    ) -> None:
        assert len(wrappers) > 0
        wrapper0 = wrappers[0]
        self.int_type = int_nmgr.int_type
        self.atm, self.bas, self.env = wrapper0.atm_bas_env
        self.wrapper0 = wrapper0
        self.int_nmgr = int_nmgr
        self.wrapper_uniqueness = _get_uniqueness([id(w) for w in wrappers])
        self.dd = {"device": wrapper0.device, "dtype": wrapper0.dtype}

        # only use hermitian argument if it is not a derivative
        self.hermitian = hermitian if int_nmgr.order == 0 else False

        # get the operator
        opname = int_nmgr.get_intgl_name(wrapper0.spherical)
        self.op = getattr(CINT, opname)
        self.optimizer = _get_intgl_optimizer(
            opname, self.atm, self.bas, self.env
        )

        # prepare the output
        comp_shape = int_nmgr.get_intgl_components_shape()
        self.outshape = comp_shape + tuple(w.nao() for w in wrappers)
        self.ncomp = reduce(operator.mul, comp_shape, 1)
        self.shls_slice = sum((w.shell_idxs for w in wrappers), ())
        self.integral_done = False

    def calc(self) -> Tensor:
        """
        Calculate the integral.

        Returns
        -------
        Tensor
            Integral tensor.

        Raises
        ------
        ValueError
            If the integral type is unknown.
        """
        assert not self.integral_done
        self.integral_done = True

        if self.int_type in ("int1e", "int2c2e"):
            return self._int2c()

        raise ValueError(f"Unknown integral type: {self.int_type}.")

    def _int2c(self) -> Tensor:
        """
        Calculate the 2-centre integrals with libcint.

        Returns
        -------
        Tensor
            Integral tensor.
        """
        drv = CGTO.GTOint2c
        outshape = self.outshape
        out = np.empty(
            (*outshape[:-2], outshape[-1], outshape[-2]), dtype=np.float64
        )
        drv(
            self.op,
            out.ctypes.data_as(ctypes.c_void_p),
            int2ctypes(self.ncomp),
            int2ctypes(self.hermitian),
            (ctypes.c_int * len(self.shls_slice))(*self.shls_slice),
            np2ctypes(self.wrapper0.full_shell_to_aoloc),
            self.optimizer,
            np2ctypes(self.atm),
            int2ctypes(self.atm.shape[0]),
            np2ctypes(self.bas),
            int2ctypes(self.bas.shape[0]),
            np2ctypes(self.env),
        )

        out = np.swapaxes(out, -2, -1)
        # TODO: check if we need to do the lines below for 3rd order grad and higher
        # if out.ndim > 2:
        #     out = np.moveaxis(out, -3, 0)
        return numpy_to_tensor(out, **self.dd)


class _CintoptHandler(ctypes.c_void_p):
    """
    Handler for the libcint optimizer.
    """

    def __del__(self):
        try:
            CGTO.CINTdel_optimizer(ctypes.byref(self))
        except AttributeError:
            pass


def _get_intgl_optimizer(
    opname: str, atm: np.ndarray, bas: np.ndarray, env: np.ndarray
) -> ctypes.c_void_p:
    """
    Get the optimizer for the integrals.

    Parameters
    ----------
    opname : str
        Name of the integral operator.
    atm : np.ndarray
        Atomic information.
    bas : np.ndarray
        Basis information.
    env : np.ndarray
        Environment information.

    Returns
    -------
    ctypes.c_void_p
        Optimizer for the integrals.
    """
    # setup the optimizer
    cintopt = ctypes.POINTER(ctypes.c_void_p)()
    optname = opname.replace("_cart", "").replace("_sph", "") + "_optimizer"
    copt = getattr(CINT, optname)
    copt(
        ctypes.byref(cintopt),
        np2ctypes(atm),
        int2ctypes(atm.shape[0]),
        np2ctypes(bas),
        int2ctypes(bas.shape[0]),
        np2ctypes(env),
    )
    opt = ctypes.cast(cintopt, _CintoptHandler)
    return opt


############### name derivation manager functions ###############


def _get_uniqueness(a: list) -> list[int]:
    """
    Get the uniqueness pattern from the list.

    Note that this does not sort the list before getting the uniqueness pattern.
    For this reason, PyTorch's `torch.unique` does not yield the same result if
    the first element is not the smallest. PyTorch even ignores the `sorted`
    argument, i.e., `torch.unique(x, return_inverse=True, sorted=False)` is
    equivalent to `torch.unique(x, return_inverse=True, sorted=True)`.
    See https://github.com/pytorch/pytorch/issues/105742.

    Parameters
    ----------
    a : list
        List of elements.

    Returns
    -------
    list[int]
        Uniqueness pattern.

    Examples
    --------
    >>> _get_uniqueness([1, 1, 2, 3, 2])
    [0, 0, 1, 2, 1]
    >>> _get_uniqueness([2, 2, 1, 3, 2])
    [0, 0, 1, 2, 0]
    """
    # Do we need to preserve the order or can we use PyTorch's `torch.unique`,
    # which sorts the result?
    # return torch.unique(torch.tensor(a), return_inverse=True, sorted=False)[1].tolist()
    s: dict = {}
    res: list[int] = []
    i = 0
    for elmt in a:
        if elmt in s:
            res.append(s[elmt])
        else:
            s[elmt] = i
            res.append(i)
            i += 1
    return res
