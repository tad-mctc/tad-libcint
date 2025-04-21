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
#
# mamba create -n wheel-312 --yes python=3.12 auditwheel "c-compiler=1.5.2" "fortran-compiler=1.5.2" cmake meson numpy pip python-build pkgconfig patchelf "sysroot_linux-64=2.12" unzip wheel

# Trap SIGINT and SIGTERM signals to exit the whole script
function handle_exit
    echo "Exited!"
    exit
end
trap handle_exit SIGINT SIGTERM SIGHUP

set -l VERSIONS "38" "39" "310" "311" "312"

for VERSION in $VERSIONS
    if ! conda env list | grep "wheel-$VERSION" >/dev/null 2>/dev/null
        echo "Creating wheel-$VERSION"

        set -l python_version (string replace -r '^(.)(.*)$' '$1.$2' $VERSION)
        mamba create -n "wheel-$VERSION" --yes "python=$python_version" auditwheel "c-compiler=1.5.2" "fortran-compiler=1.5.2" cmake "libgcc<14" "libgfortran<14" "libstdcxx<14" meson numpy pip python-build pkgconfig patchelf "sysroot_linux-64=2.12" unzip wheel
    end

    if test -f "wheels/tad_libcint-0.1.1-cp$VERSION-cp$VERSION-manylinux_2_12_x86_64.manylinux2010_x86_64.whl"
        echo "Skipping wheel-$VERSION"
        continue
    end

    conda activate wheel-$VERSION
    echo "Activate wheel-$VERSION"
    python -m build --wheel
    auditwheel repair -w wheels --plat manylinux_2_12_x86_64 dist/*-cp$VERSION-cp$VERSION-linux_x86_64.whl
    conda deactivate
    rm -rf build/ dist/

    echo ""
    echo ""
end
