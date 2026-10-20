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
Test import of libraries.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import tad_libcint
from tad_libcint.library import (
    ATM_SLOTS,
    BAS_SLOTS,
    PTR_ENV_START,
    int2c,
    int2c_lattice,
)


def test_load() -> None:
    """The symbols used by the package are exported."""
    assert tad_libcint.CGTO.GTOint2c.__name__ == "GTOint2c"
    assert tad_libcint.CINT.int1e_ovlp_sph.__name__ == "int1e_ovlp_sph"
    assert tad_libcint.CINT.CINTdel_optimizer.__name__ == "CINTdel_optimizer"


def test_unique_library_names() -> None:
    """The libraries are suffixed and do not shadow PySCF's libraries."""
    suffix = tad_libcint.CINT.path.suffix
    assert suffix == (".dll" if sys.platform == "win32" else ".so")
    for lib in (tad_libcint.CINT, tad_libcint.CGTO):
        assert lib.path.name.endswith(f"_tad{suffix}")
        assert lib.path.is_file()

    directory = Path(tad_libcint.__file__).parent
    names = {p.name for p in directory.glob(f"lib*{suffix}*")}
    assert names == {f"libcint_tad{suffix}", f"libcgto_tad{suffix}"}


@pytest.mark.parametrize("component", [-1, 3])
def test_int2c_components_out_of_range(component: int) -> None:
    """Components beyond those of the integral are refused."""
    with pytest.raises(ValueError, match="out of 3"):
        int2c(
            "int1e_r_sph",
            (3,),
            False,
            shls_slice=(0, 1, 0, 1),
            components=[0, component],
            atm=np.zeros((1, ATM_SLOTS), dtype=np.int32),
            bas=np.zeros((1, BAS_SLOTS), dtype=np.int32),
            env=np.zeros(PTR_ENV_START),
            ao_loc=np.array([0, 1], dtype=np.int32),
        )


@pytest.mark.parametrize(
    "translations, image, match",
    [
        (np.zeros((2, 3)), 0, "translations of shape"),
        (np.zeros((1, 3)), 1, "out of 1 images"),
    ],
)
def test_int2c_lattice_invalid(
    translations: np.ndarray, image: int, match: str
) -> None:
    """Translations that do not match the weights, and image indices beyond
    them, are refused."""
    with pytest.raises(ValueError, match=match):
        int2c_lattice(
            "int1e_ovlp_sph",
            (),
            shls_slice=(0, 1, 1, 2),
            pairs=np.array([[0, 1, image]]),
            translations=translations,
            weights=np.ones((1, 1)),
            atm=np.zeros((2, ATM_SLOTS), dtype=np.int32),
            bas=np.zeros((2, BAS_SLOTS), dtype=np.int32),
            env=np.zeros(PTR_ENV_START),
            ao_loc=np.array([0, 1, 2], dtype=np.int32),
        )


_TAD = """
import torch
from tad_mctc.io.structure import Structure
from tad_libcint import Basis, Shell, overlap
dd = {"dtype": torch.double}
alpha, coeff = torch.tensor([1.0], **dd), torch.tensor([2.5], **dd)
shells = {1: [Shell(angmom=0, alphas=alpha, coeffs=coeff)]}
basis = Basis(shells=shells)
positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], **dd)
overlap(basis, Structure(numbers=torch.tensor([1, 1]), positions=positions))
"""

_PYSCF = """
from pyscf import gto, scf
mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
scf.RHF(mol).kernel()
"""


@pytest.mark.parametrize("order", ["tad-first", "pyscf-first"])
def test_coexistence_with_pyscf(order: str) -> None:
    """Both packages work in one process, independent of the loading order."""
    if importlib.util.find_spec("pyscf") is None:
        pytest.skip("PySCF is not installed")

    script = _TAD + _PYSCF if order == "tad-first" else _PYSCF + _TAD
    subprocess.run([sys.executable, "-c", script], check=True)


# PyTorch runs its OpenMP runtime first, then lists the loaded OpenMP runtimes
_OPENMP = (
    """
import torch
torch.set_num_threads(2)
torch.rand(2**22).exp().sum()
"""
    + _TAD
    + """
import ctypes, os
dyld = ctypes.CDLL(None)
dyld._dyld_get_image_name.restype = ctypes.c_char_p
images = [dyld._dyld_get_image_name(i) for i in range(dyld._dyld_image_count())]
print(*{os.path.realpath(i.decode()) for i in images if i.endswith(b"/libomp.dylib")})
"""
)


@pytest.mark.skipif(sys.platform != "darwin", reason="LLVM's libomp of macOS")
def test_openmp_runtime_of_pytorch() -> None:
    """The libraries share PyTorch's libomp, a second one would abort."""
    result = subprocess.run(
        [sys.executable, "-c", _OPENMP],
        check=True,
        capture_output=True,
        text=True,
    )
    (runtime,) = result.stdout.split()
    assert Path(runtime).is_relative_to(Path(torch.__file__).parent.resolve())
