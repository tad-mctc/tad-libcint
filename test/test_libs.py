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
import tad_libcint


def test_load() -> None:
    """Try loading the libraries."""
    cgto = tad_libcint.CGTO
    cint = tad_libcint.CINT

    # test loading the functions
    gto1 = getattr(cgto, "GTOval_cart")
    gto2 = getattr(cgto, "GTOval_ip_cart")
    cint1 = getattr(cint, "int1e_ovlp_sph")
    cint2 = getattr(cint, "int1e_rrkin_sph")

    assert gto1.__name__ == "GTOval_cart"
    assert gto2.__name__ == "GTOval_ip_cart"
    assert cint1.__name__ == "int1e_ovlp_sph"
    assert cint2.__name__ == "int1e_rrkin_sph"
