#!/usr/bin/env fish

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


# environments are created with:
# mamba create -n wheel-<version> --yes python=<version> auditwheel c-compiler fortran-compiler cmake meson numpy pip python-build pkgconfig patchelf unzip wheel

set -l VERSIONS "38" "39" "310" "311" "312"

# Trap SIGINT and SIGTERM signals to exit the whole script
function handle_exit
    echo "Exited!"
    exit
end
trap handle_exit SIGINT SIGTERM

for VERSION in $VERSIONS
    conda activate wheel-$VERSION
    echo "Activate wheel-$VERSION"
    python -m build --wheel
    auditwheel repair -w wheels --plat manylinux_2_12_x86_64 dist/*-cp$VERSION-cp$VERSION-linux_x86_64.whl
    conda deactivate
    rm -rf build/ dist/

    echo ""
    echo ""
end
