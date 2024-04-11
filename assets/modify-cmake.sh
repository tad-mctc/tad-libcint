#!/bin/bash

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


# This script modifies the main CMakeList.txt in tad-libcint/libs.
#
# Note additional modifications are required in the CMakeFiles of the remaining
# subpackages in tad-libcint/libs. Use the following:
#
# set_target_properties(cgto PROPERTIES
#   #LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
#   COMPILE_FLAGS ${OpenMP_C_FLAGS}
#   LINK_FLAGS ${OpenMP_C_FLAGS}
# )
#
#
# Always execute in root of repository!

DIR="tad-libcint"

if [[ "$(basename "$(pwd)")" == "$DIR" ]]; then
    CM="tad-libcint/libs/CMakeLists.txt"

    sed -i 's/-DWITH_RANGE_COULOMB:STRING=1/-DWITH_RANGE_COULOMB:STRING=0/' $CM

    sed -i 's/-DBUILD_MARCH_NATIVE:STRING=${BUILD_MARCH_NATIVE}/-DBUILD_MARCH_NATIVE:STRING=${BUILD_MARCH_NATIVE}\n        -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/' $CM

    sed -i 's/add_dependencies(ao2mo libcint)/#NO_DXTB add_dependencies(ao2mo libcint)/' $CM

    sed -i 's/add_subdirectory(ao2mo)/#NO_DXTB add_subdirectory(ao2mo)/' $CM
    sed -i 's/add_subdirectory(mcscf)/#NO_DXTB add_subdirectory(mcscf)/' $CM
    sed -i 's/add_subdirectory(cc)/#NO_DXTB add_subdirectory(cc)/' $CM
    sed -i 's/add_subdirectory(ri)/#NO_DXTB add_subdirectory(ri)/' $CM
    sed -i 's/add_subdirectory(localizer)/#NO_DXTB add_subdirectory(localizer)/' $CM
    sed -i 's/add_subdirectory(pbc)/#NO_DXTB add_subdirectory(pbc)/' $CM
    sed -i 's/add_subdirectory(agf2)/#NO_DXTB add_subdirectory(agf2)/' $CM
else
    echo "The current directory is not '$DIR'."
    exit 1
fi
