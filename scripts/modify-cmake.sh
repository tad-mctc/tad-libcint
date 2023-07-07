#!/bin/bash

# This script modifies the main CMakeList.txt in dxtblibs/libs.
#
# Note additional modifications are required in the CMakeFiles of the remaining 
# subpackages in dxtblibs/libs. Use the following:
#
# set_target_properties(cgto PROPERTIES
#   #LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
#   COMPILE_FLAGS ${OpenMP_C_FLAGS}
#   LINK_FLAGS ${OpenMP_C_FLAGS}
# )
#
#
# Always execute in root of repository!

DIR="dxtblibs"

if [[ "$(basename "$(pwd)")" == "$DIR" ]]; then
    CM="dxtblibs/libs/CMakeLists.txt"
    
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