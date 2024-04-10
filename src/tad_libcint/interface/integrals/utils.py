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
Integrals: Utility
==================

This module contains utility functions for the integral interface.
"""
from __future__ import annotations

import copy

import torch

from tad_libcint.typing import Callable, Tensor

from ..namemanager import IntorNameManager
from ..wrapper import LibcintWrapper

__all__ = ["get_integrals", "gather_at_dims"]


def get_integrals(
    int_nmgrs: list[IntorNameManager],
    wrappers: list[LibcintWrapper],
    int_fcn: Callable[[list[LibcintWrapper], IntorNameManager], Tensor],
    new_axes_pos: list[int | None],
) -> list[Tensor]:
    # Return the list of tensors of the integrals given by the list of integral
    # names. Int_fcn is the integral function that receives the name and
    # returns the results. If new_axes_pos is specified, then move the new axes
    # to 0, otherwise, just leave it as it is.

    res: list[Tensor] = []
    # indicating if the integral is available in the libcint-generated file
    int_avail: list[bool] = [False] * len(int_nmgrs)

    for i in range(len(int_nmgrs)):
        res_i: Tensor | None = None

        # check if the integral can be calculated from the previous results
        for j in range(i - 1, -1, -1):
            # check the integral names equivalence
            transpose_path = int_nmgrs[j].get_transpose_path_to(int_nmgrs[i])
            if transpose_path is not None:
                # if the swapped wrappers remain unchanged, then just use the
                # transposed version of the previous version
                # TODO: think more about this (do we need to use different
                # transpose path? e.g. transpose_path[::-1])
                twrappers = _swap_list(wrappers, transpose_path)
                if twrappers == wrappers:
                    res_i = _transpose(res[j], transpose_path)
                    permute_path = int_nmgrs[j].get_comp_permute_path(transpose_path)
                    res_i = res_i.permute(*permute_path)
                    break

                # otherwise, use the swapped integral with the swapped wrappers,
                # only if the integral is available in the libcint-generated
                # files
                elif int_avail[j]:
                    res_i = int_fcn(twrappers, int_nmgrs[j])
                    res_i = _transpose(res_i, transpose_path)
                    permute_path = int_nmgrs[j].get_comp_permute_path(transpose_path)
                    res_i = res_i.permute(*permute_path)
                    break

                # if the integral is not available, then continue the searching
                else:
                    continue

        if res_i is None:
            try:
                # successfully executing the line below indicates that the integral
                # is available in the libcint-generated files
                res_i = int_fcn(wrappers, int_nmgrs[i])
            except AttributeError as e:
                msg = f"The integral {int_nmgrs[i].fullname} is not available from libcint, please add it"

                raise AttributeError(msg) from e

            int_avail[i] = True

        res.append(res_i)

    # move the new axes (if any) to dimension 0
    assert res_i is not None
    for i in range(len(res)):
        new_axes_pos_i = new_axes_pos[i]
        if new_axes_pos_i is not None:
            res[i] = torch.movedim(res[i], new_axes_pos_i, 0)

    return res


def _transpose(a: Tensor, axes: list[tuple[int, int]]) -> Tensor:
    # perform the transpose of two axes for tensor a
    for axis2 in axes:
        a = a.transpose(*axis2)
    return a


def _swap_list(a: list, swaps: list[tuple[int, int]]) -> list:
    # swap the elements according to the swaps input
    res = copy.copy(a)  # shallow copy
    for idxs in swaps:
        res[idxs[0]], res[idxs[1]] = res[idxs[1]], res[idxs[0]]  # swap the elements
    return res


def gather_at_dims(inp: Tensor, mapidxs: list[Tensor], dims: list[int]) -> Tensor:
    # expand inp in the dimension dim by gathering values based on the given
    # mapping indices

    # mapidx: (nnew,) with value from 0 to nold - 1
    # inp: (..., nold, ...)
    # out: (..., nnew, ...)
    out = inp
    for dim, mapidx in zip(dims, mapidxs):
        if dim < 0:
            dim = out.ndim + dim
        map2 = mapidx[(...,) + (None,) * (out.ndim - 1 - dim)]
        map2 = map2.expand(*out.shape[:dim], -1, *out.shape[dim + 1 :])
        out = torch.gather(out, dim=dim, index=map2)
    return out
